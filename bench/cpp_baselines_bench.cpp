// Euler wrapper around the unchanged Davidson solver. Build with C++17/OpenMP.
// Usage: cpp_baselines_bench n input.bin serial|openmp --reference ref.bin
//        --samples samples.jsonl --cores p --cpus cpu0,cpu1 --config-id id
//        [--min-time 1 --min-runs 5 --max-runs 200] [--smoke]
// Only Davidson() is timed. Input, independent checks, affinity inspection and
// flushed JSONL output are outside that interval. Existing files are never replaced.

#include "davidson.h"
#include "matrix.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sched.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
constexpr int kRoots = 4, kMaxIterations = 100;
constexpr double kTolerance = 1e-8;
const double unavailable = std::numeric_limits<double>::quiet_NaN();
using Clock = std::chrono::steady_clock;

std::string json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else out << c;
    }
    return out.str() + '"';
}
std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}
std::string boolean(bool value) { return value ? "true" : "false"; }
template <typename T> std::string array(const std::vector<T>& values) {
    std::string out = "[";
    for (std::size_t i = 0; i < values.size(); ++i)
        out += (i ? "," : "") + number(values[i]);
    return out + "]";
}
struct Object : std::map<std::string, std::string> {
    using std::map<std::string, std::string>::map;
    std::string json() const {
        std::string out = "{";
        for (const auto& field : *this)
            out += (out.size() > 1 ? "," : "") + json_string(field.first) + ":" + field.second;
        return out + "}";
    }
};
std::string environment(const char* name) {
    const char* value = std::getenv(name);
    return value ? json_string(value) : "null";
}
std::string utc_now() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&now, &tm);
    char buffer[32];
    std::strftime(buffer, sizeof buffer, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buffer;
}

struct Config {
    int n = 0, cores = 0, min_runs = 5, max_runs = 200;
    double min_time = 1;
    bool smoke = false;
    std::string input, reference, samples, backend, id;
    std::vector<int> cpus;
};
int integer(const std::string& text, int low, int high) {
    std::size_t used = 0;
    const long value = std::stol(text, &used);
    if (used != text.size() || value < low || value > high)
        throw std::invalid_argument("invalid integer: " + text);
    return static_cast<int>(value);
}
Config parse(int argc, char** argv) {
    if (argc < 4) throw std::invalid_argument("expected n input.bin serial|openmp and required options");
    Config c;
    c.n = integer(argv[1], 24, 32768);  // Common suite minimum: aligned GD subspace dimension.
    c.input = argv[2];
    c.backend = argv[3];
    if (c.backend != "serial" && c.backend != "openmp")
        throw std::invalid_argument("backend must be serial or openmp");
    std::set<std::string> seen;
    for (int i = 4; i < argc; ++i) {
        const std::string flag = argv[i];
        if (!seen.insert(flag).second) throw std::invalid_argument("duplicate option: " + flag);
        if (flag == "--smoke") { c.smoke = true; continue; }
        if (++i == argc) throw std::invalid_argument("missing value for " + flag);
        const std::string value = argv[i];
        if (flag == "--reference") c.reference = value;
        else if (flag == "--samples") c.samples = value;
        else if (flag == "--config-id") c.id = value;
        else if (flag == "--cores") c.cores = integer(value, 1, CPU_SETSIZE);
        else if (flag == "--min-runs") c.min_runs = integer(value, 1, 1000000);
        else if (flag == "--max-runs") c.max_runs = integer(value, 1, 200);
        else if (flag == "--min-time") {
            std::size_t used = 0;
            c.min_time = std::stod(value, &used);
            if (used != value.size() || !std::isfinite(c.min_time) || c.min_time < 0)
                throw std::invalid_argument("--min-time must be finite and nonnegative");
        } else if (flag == "--cpus") {
            if (value.empty() || value.back() == ',') throw std::invalid_argument("empty CPU identifier");
            std::istringstream stream(value);
            std::string cpu;
            while (std::getline(stream, cpu, ',')) c.cpus.push_back(integer(cpu, 0, CPU_SETSIZE - 1));
        } else throw std::invalid_argument("unknown option: " + flag);
    }
    if (c.reference.empty() || c.samples.empty() || c.id.empty() || c.cores == 0)
        throw std::invalid_argument("--reference, --samples, --config-id and --cores are required");
    const std::set<int> unique(c.cpus.begin(), c.cpus.end());
    if (static_cast<int>(unique.size()) != c.cores || unique.size() != c.cpus.size())
        throw std::invalid_argument("--cpus must contain exactly --cores distinct CPU identifiers");
    if (c.backend == "serial" && c.cores != 1)
        throw std::invalid_argument("serial requires exactly one core");
    if (c.smoke && c.max_runs > 3)
        throw std::invalid_argument("smoke mode permits at most three measured solves");
#ifndef _OPENMP
    if (c.backend == "openmp") throw std::invalid_argument("OpenMP backend requires a build with -fopenmp");
#endif
    return c;
}

std::vector<double> read_binary(const std::string& path, std::size_t count) {
    if (std::filesystem::file_size(path) != count * sizeof(double))
        throw std::runtime_error("incorrect binary length: " + path);
    std::ifstream input(path, std::ios::binary);
    std::vector<double> data(count);
    if (!input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(double))))
        throw std::runtime_error("cannot read binary input: " + path);
    if (!std::all_of(data.begin(), data.end(), [](double x) { return std::isfinite(x); }))
        throw std::runtime_error("nonfinite binary value: " + path);
    return data;
}
std::string checksum(const std::vector<double>& data) {
    std::uint64_t sum = 0, position = 0;
    for (double value : data) {
        std::uint64_t word;
        std::memcpy(&word, &value, sizeof word);
        sum += word * (2 * position++ + 1);
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << sum;
    return out.str();
}
std::vector<int> affinity() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof mask, &mask) != 0)
        return {};  // An empty mask fails the resource gate below.
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
    return cpus;
}
struct Resources {
    std::vector<std::vector<int>> masks;
    std::vector<int> current_cpus;
    bool ok = false, dynamic = false;
    Object json;
};
Resources inspect_resources(const Config& c) {
    Resources r;
    if (c.backend == "openmp") {
#ifdef _OPENMP
        r.dynamic = omp_get_dynamic() != 0;
#pragma omp parallel
        {
#pragma omp single
            { r.masks.resize(omp_get_num_threads()); r.current_cpus.resize(omp_get_num_threads()); }
            const int thread = omp_get_thread_num();
            r.masks[thread] = affinity();
            r.current_cpus[thread] = sched_getcpu();
        }
#endif
    } else {
        r.masks.push_back(affinity());
        r.current_cpus.push_back(sched_getcpu());
    }
    std::vector<int> observed;
    r.ok = !r.dynamic && static_cast<int>(r.masks.size()) == c.cores;
    std::string threads = "[";
    for (std::size_t i = 0; i < r.masks.size(); ++i) {
        r.ok = r.ok && r.masks[i].size() == 1 && r.masks[i][0] == r.current_cpus[i];
        observed.insert(observed.end(), r.masks[i].begin(), r.masks[i].end());
        threads += (i ? "," : "") + Object{{"thread", number(i)}, {"cpus", array(r.masks[i])},
                                           {"current_cpu", number(r.current_cpus[i])}}.json();
    }
    auto expected = c.cpus;
    std::sort(expected.begin(), expected.end());
    std::sort(observed.begin(), observed.end());
    r.ok = r.ok && observed == expected;
    r.json = {{"cores", number(c.cores)}, {"ranks", "1"}, {"omp_threads", number(r.masks.size())},
              {"blas_threads", "1"}, {"uses_blas", "false"}, {"selected_cpus", array(c.cpus)},
              {"main_thread_affinity", array(affinity())}, {"thread_affinities", threads + "]"},
              {"omp_dynamic", boolean(r.dynamic)}, {"affinity_pass", boolean(r.ok)}, {"ok", boolean(r.ok)},
              {"measurement", json_string("separate OpenMP team immediately after each solve, outside timer")}};
    return r;
}

struct Sample {
    double elapsed = unavailable, eigen_error = unavailable, residual = unavailable, orth_error = unavailable;
    int iterations = -1, matvecs = -1;
    bool converged = false, strict_gate = false, gate = false;
    std::vector<double> eigenvalues, errors, residuals;
    std::string failure;
    Object resources;
};
Sample solve(const Config& c, const Matrix& A, const Matrix& guess, const std::vector<double>& reference) {
    Sample s;
    const auto begin = Clock::now();
    try {
        const DavidsonResult result = Davidson(A, guess, kRoots, kTolerance, kMaxIterations,
                                               c.backend == "serial" ? Backend::Serial : Backend::OpenMP);
        s.elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
        s.iterations = result.iterations;
        // Each accepted basis vector receives one A*v; no restart or unused
        // final expansion exists on any normal return of this implementation.
        s.matvecs = result.subspace_size;
        s.converged = result.converged;
        s.eigenvalues = result.eigenvalues;
        const Matrix& X = result.eigenvectors;
        if (s.eigenvalues.size() != kRoots || X.rows() != c.n || X.cols() != kRoots)
            throw std::runtime_error("solver did not return four eigenpairs");
        bool finite = true;
        s.eigen_error = s.residual = 0;
        double orth_squared = 0;
        for (int k = 0; k < kRoots; ++k) {
            const double error = std::abs(s.eigenvalues[k] - reference[k]);
            s.errors.push_back(error);
            finite = finite && std::isfinite(error);
            s.eigen_error = std::max(s.eigen_error, error);
            double residual_squared = 0;
            for (int i = 0; i < c.n; ++i) {
                finite = finite && std::isfinite(X[i][k]);
                double ax = 0;
                for (int j = 0; j < c.n; ++j) ax += A[i][j] * X[j][k];
                const double difference = ax - s.eigenvalues[k] * X[i][k];
                residual_squared += difference * difference;
            }
            const double residual = std::sqrt(residual_squared);
            finite = finite && std::isfinite(residual);
            s.residuals.push_back(residual);
            s.residual = std::max(s.residual, residual);
            for (int l = 0; l < kRoots; ++l) {
                double inner = 0;
                for (int i = 0; i < c.n; ++i) inner += X[i][k] * X[i][l];
                const double difference = inner - (k == l ? 1.0 : 0.0);
                orth_squared += difference * difference;
            }
        }
        s.orth_error = std::sqrt(orth_squared);
        finite = finite && std::isfinite(s.orth_error);
        if (!finite) s.eigen_error = s.residual = s.orth_error = unavailable;
        s.strict_gate = finite && s.converged && s.eigen_error < kTolerance &&
                        s.residual < kTolerance && s.orth_error < kTolerance;
        if (!s.strict_gate) s.failure = "four-pair convergence/finite/eigenvalue/residual/orthogonality gate failed";
    } catch (const std::exception& error) {
        if (!std::isfinite(s.elapsed)) s.elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
        s.failure = error.what();
    }
    const Resources resources = inspect_resources(c);
    s.resources = resources.json;
    s.gate = s.strict_gate && resources.ok;
    if (!resources.ok) s.failure += (s.failure.empty() ? "" : "; ") + std::string("actual thread count/affinity differs from requested CPUs");
    return s;
}
double median(std::vector<double> values) {
    if (values.empty()) return unavailable;
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
}
void write_line(std::FILE* file, const Object& record) {
    const std::string line = record.json() + "\n";
    if (std::fwrite(line.data(), 1, line.size(), file) != line.size() || std::fflush(file) != 0)
        throw std::runtime_error("cannot write or flush samples JSONL");
}

int run(const Config& c, std::FILE* raw) {
    const std::uint16_t endian = 1;
    if (sizeof(double) != 8 || !std::numeric_limits<double>::is_iec559 ||
        *reinterpret_cast<const unsigned char*>(&endian) != 1)
        throw std::runtime_error("dump format requires native little-endian IEEE float64");
    const auto data = read_binary(c.input, std::size_t(c.n) * (c.n + kRoots));
    const auto reference = read_binary(c.reference, kRoots);
    if (!std::is_sorted(reference.begin(), reference.end()))
        throw std::runtime_error("reference eigenvalues must be sorted ascending");
    Matrix A(c.n, c.n), guess(c.n, kRoots);
    for (int i = 0; i < c.n; ++i) {
        for (int j = 0; j < c.n; ++j) A[i][j] = data[std::size_t(i) * c.n + j];
        for (int k = 0; k < kRoots; ++k) guess[i][k] = data[std::size_t(c.n) * c.n + i * kRoots + k];
    }
    for (int i = 0; i < c.n; ++i)
        for (int j = i + 1; j < c.n; ++j)
            if (A[i][j] != A[j][i]) throw std::runtime_error("input matrix is not symmetric");
    char host[256]{};
    if (gethostname(host, sizeof host - 1) != 0) throw std::runtime_error("cannot obtain hostname");
    Object versions{{"compiler", json_string(__VERSION__)}, {"cplusplus", number(__cplusplus)}};
#ifdef _OPENMP
    versions["openmp"] = number(_OPENMP);
#else
    versions["openmp"] = "null";
#endif
    Object threads;
    for (const char* name : {"OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES",
                             "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"})
        threads[name] = environment(name);
    Object settings{{"nroots", "4"}, {"tol_residual", number(kTolerance)}, {"max_iterations", "100"},
                    {"which", json_string("smallest algebraic")}, {"problem", json_string("real symmetric")},
                    {"preconditioner", json_string("per-root r_i/(theta-A_ii), signed denominator floor 1e-8")},
                    {"lindep_relative_norm", number(kLinDepTol)}, {"initial_vectors", "4"},
                    {"block_size", json_string("one correction per unconverged root")}, {"restart", "false"},
                    {"subspace_limit", json_string("no explicit limit")}, {"residual_slack", "1"},
                    {"timed_region", json_string("Davidson call only, including existing internal profiling")},
                    {"thread_environment", threads.json()}};
    Object common{{"schema_version", "1"}, {"config_id", json_string(c.id)}, {"n", number(c.n)},
                  {"k", "4"}, {"nroots", "4"}, {"solver", json_string("cpp_" + c.backend)},
                  {"backend", json_string("cpp_" + c.backend)},
                  {"cores", number(c.cores)}, {"ranks", "1"}, {"omp_threads", number(c.cores)}, {"blas_threads", "1"},
                  {"hostname", json_string(host)}, {"job_id", environment("SLURM_JOB_ID")},
                  {"source_revision", environment("BASELINES_SOURCE_REVISION")},
                  {"input_checksum", json_string(checksum(data))}, {"input_path", json_string(c.input)},
                  {"reference_path", json_string(c.reference)}, {"reference_eigenvalues", array(reference)},
                  {"raw_path", json_string(c.samples)}, {"smoke", boolean(c.smoke)},
                  {"settings", settings.json()}, {"versions", versions.json()},
                  {"iteration_definition", json_string("projection/solve/residual cycles")},
                  {"matvec_count_method", json_string("final basis size: one A*v per accepted basis vector; no restart")},
                  {"residual_slack", "1"}};
    std::vector<double> times, iteration_counts, matvec_counts;
    double total = 0, worst_eigen = 0, worst_residual = 0, worst_orth = 0;
    bool all_ok = true, all_converged = true;
    std::string failure, stop_reason;
    Object last_resources;
    int warmups = 0;
    for (int index = 0; ; ++index) {
        const Sample s = solve(c, A, guess, reference);
        const bool warmup = index == 0;
        Object record = common;
        record["record_type"] = json_string("sample");
        record["sample_index"] = number(index);
        record["sample_id"] = json_string(c.id + ":" + std::to_string(index));
        record["utc"] = json_string(utc_now());
        record["phase"] = json_string(warmup ? "warmup" : "measured");
        record["elapsed_s"] = number(s.elapsed);
        record["iterations"] = s.iterations < 0 ? "null" : number(s.iterations);
        record["matvecs"] = s.matvecs < 0 ? "null" : number(s.matvecs);
        record["converged"] = boolean(s.converged);
        record["eigenvalues"] = array(s.eigenvalues);
        record["eigenvalue_errors"] = array(s.errors);
        record["residuals"] = array(s.residuals);
        record["max_eigenvalue_error"] = number(s.eigen_error);
        record["max_residual"] = number(s.residual);
        record["orth_error"] = number(s.orth_error);
        record["numerical_gate_pass"] = boolean(s.strict_gate);
        record["strict_gate_pass"] = boolean(s.gate);
        record["gate_pass"] = record["ok"] = boolean(s.gate);
        record["failure_reason"] = s.failure.empty() ? "null" : json_string(s.failure);
        record["resources"] = s.resources.json();
        write_line(raw, record);
        last_resources = s.resources;
        all_ok = all_ok && s.gate;
        all_converged = all_converged && s.converged;
        worst_eigen = std::isfinite(s.eigen_error) ? std::max(worst_eigen, s.eigen_error) : unavailable;
        worst_residual = std::isfinite(s.residual) ? std::max(worst_residual, s.residual) : unavailable;
        worst_orth = std::isfinite(s.orth_error) ? std::max(worst_orth, s.orth_error) : unavailable;
        if (warmup) ++warmups;
        else {
            times.push_back(s.elapsed);
            total += s.elapsed;
            if (s.iterations >= 0) iteration_counts.push_back(s.iterations);
            if (s.matvecs >= 0) matvec_counts.push_back(s.matvecs);
        }
        if (!s.gate) { failure = s.failure; stop_reason = "failed_gate"; break; }
        if (static_cast<int>(times.size()) >= c.min_runs && total >= c.min_time) {
            stop_reason = "targets_met"; break;
        }
        if (static_cast<int>(times.size()) >= c.max_runs) { stop_reason = "max_runs"; break; }
    }
    Object summary = common;
    summary["record_type"] = json_string("summary");
    summary["utc"] = json_string(utc_now());
    summary["warmups"] = number(warmups);
    summary["runs"] = number(times.size());
    summary["min_runs"] = number(c.min_runs);
    summary["max_runs"] = number(c.max_runs);
    summary["min_time_s"] = number(c.min_time);
    summary["times"] = array(times);
    summary["total_timed_s"] = number(total);
    summary["minimum_time_met"] = boolean(total >= c.min_time);
    summary["minimum_runs_met"] = boolean(static_cast<int>(times.size()) >= c.min_runs);
    summary["stop_reason"] = json_string(stop_reason);
    summary["median"] = number(median(times));
    summary["median_ms"] = number(1000 * median(times));
    summary["min_ms"] = times.empty() ? "null" : number(1000 * *std::min_element(times.begin(), times.end()));
    summary["max_ms"] = times.empty() ? "null" : number(1000 * *std::max_element(times.begin(), times.end()));
    summary["iterations_median"] = number(median(iteration_counts));
    summary["matvecs_median"] = number(median(matvec_counts));
    summary["max_eigenvalue_error"] = number(worst_eigen);
    summary["max_residual"] = number(worst_residual);
    summary["max_orth_error"] = number(worst_orth);
    summary["all_converged"] = boolean(all_converged);
    summary["strict_gate_pass"] = boolean(all_ok);
    summary["gate_pass"] = summary["ok"] = boolean(all_ok);
    summary["failure_reason"] = failure.empty() ? "null" : json_string(failure);
    summary["resources"] = last_resources.json();
    std::cout << summary.json() << '\n';
    if (!all_ok) std::cerr << "FAIL: " << failure << '\n';
    return all_ok ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
    Config config;
    std::FILE* samples = nullptr;
    int status = 1;
    try {
        config = parse(argc, argv);
        samples = std::fopen(config.samples.c_str(), "wx");
        if (!samples) throw std::runtime_error("cannot create fresh samples file: " + config.samples + ": " + std::strerror(errno));
        status = run(config, samples);
    } catch (const std::exception& error) {
        Object record{{"schema_version", "1"}, {"record_type", json_string("error")}, {"config_id", json_string(config.id)},
                      {"solver", json_string("cpp_" + config.backend)}, {"n", number(config.n)},
                      {"ok", "false"}, {"gate_pass", "false"}, {"failure_reason", json_string(error.what())}};
        if (samples) {
            try { write_line(samples, record); } catch (const std::exception&) {}
        }
        std::cout << record.json() << '\n';
        std::cerr << "FAIL: " << error.what() << '\n';
    }
    if (samples && std::fclose(samples) != 0) {
        std::cerr << "FAIL: cannot close samples JSONL\n";
        return 1;
    }
    return status;
}

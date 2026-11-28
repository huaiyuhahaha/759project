#include "Settings.h"

#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

enum class InputKey {
    latVec, atoms, functional, outputfile, totalEConv, maxIter, davidsonConv, unknown
};

InputKey input_key(const std::string& name) {
    static const std::unordered_map<std::string, InputKey> keys{
        {"latVec", InputKey::latVec},
        {"atoms", InputKey::atoms},
        {"functional", InputKey::functional},
        {"outputfile", InputKey::outputfile},
        {"totalEConv", InputKey::totalEConv},
        {"maxIter", InputKey::maxIter},
        {"davidsonConv", InputKey::davidsonConv}
    };
    const auto found = keys.find(name);
    return found == keys.end() ? InputKey::unknown : found->second;
}

Matrix<double> identity_lattice() {
    Matrix<double> lattice(3, 3);
    for (int i = 0; i < 3; ++i)
        lattice[i][i] = 1.0;
    return lattice;
}

} // namespace

Settings::Settings()
    : cryStruct(identity_lattice(), Matrix<double>(0, 3), {}, Matrix<double>(0, 1)),
      functional(),
      inputfile(),
      outputfile(),
      totalEConv(),
      maxIter(),
      davidsonConv()
{}

Settings::~Settings() = default;

void Settings::readInput() {
    if (inputfile.empty())
        throw std::invalid_argument("Set Settings::inputfile before calling readInput()");
    std::ifstream input(inputfile);
    if (!input)
        throw std::runtime_error("Cannot open input file: " + inputfile);

    // Parse into temporary objects so a malformed file does not change settings.
    Settings parsed;
    Matrix<double> lattice(3, 3);
    Matrix<double> coordinates(0, 3);
    Matrix<double> masses(0, 1);
    std::vector<int> numbers;
    std::vector<std::string> pseudopotentials;
    std::set<std::string> seen;
    std::string line;
    int line_number = 0;
    auto fail = [&](const std::string& message) {
        throw std::runtime_error(inputfile + ":" + std::to_string(line_number) + ": " + message);
    };
    auto next_line = [&]() {
        while (std::getline(input, line)) {
            ++line_number;
            const auto comment = line.find('#');
            if (comment != std::string::npos)
                line.erase(comment);
            if (line.find_first_not_of(" \t\r\n") != std::string::npos)
                return true;
        }
        if (input.bad())
            fail("Error reading input file");
        return false;
    };
    auto require_end = [&](std::istringstream& row) {
        std::string extra;
        if (row >> extra)
            fail("Unexpected extra value: " + extra);
    };

    while (next_line()) {
        std::istringstream row(line);
        std::string key;
        row >> key;
        if (!seen.insert(key).second)
            fail("Duplicate entry: " + key);

        auto read_value = [&]() {
            std::string value;
            if (!(row >> value))
                fail("Missing value for " + key);
            require_end(row);
            return value;
        };
        auto read_tolerance = [&]() {
            const std::string value = read_value();
            double tolerance;
            std::istringstream numeric(value);
            if (!(numeric >> tolerance) || !std::isfinite(tolerance) || tolerance <= 0)
                fail(key + " must be finite and positive");
            require_end(numeric);
            return value;
        };

        switch (input_key(key)) {
        case InputKey::latVec: {
            require_end(row);
            for (int i = 0; i < 3; ++i) {
                if (!next_line())
                    fail("latVec requires three rows");
                std::istringstream values(line);
                for (int j = 0; j < 3; ++j) {
                    if (!(values >> lattice[i][j]) || !std::isfinite(lattice[i][j]))
                        fail("Each lattice row requires three finite numbers");
                }
                require_end(values);
            }
            break;
        }
        case InputKey::atoms: {
            int count;
            if (!(row >> count) || count <= 0)
                fail("atoms requires a positive integer count");
            require_end(row);
            coordinates = Matrix<double>(count, 3);
            masses = Matrix<double>(count, 1);
            numbers.resize(count);
            pseudopotentials.resize(count);
            for (int i = 0; i < count; ++i) {
                if (!next_line())
                    fail("Fewer atom rows than requested");
                std::istringstream values(line);
                if (!(values >> numbers[i] >> masses[i][0]
                             >> coordinates[i][0] >> coordinates[i][1] >> coordinates[i][2]
                             >> pseudopotentials[i]))
                    fail("Atom row requires: atomic_number mass fractional_x fractional_y fractional_z pseudopotential");
                require_end(values);
                if (numbers[i] < 1 || numbers[i] > 118 ||
                    !std::isfinite(masses[i][0]) || masses[i][0] <= 0)
                    fail("Invalid atomic number or mass");
                for (int j = 0; j < 3; ++j)
                    if (!std::isfinite(coordinates[i][j]))
                        fail("Atomic coordinates must be finite");
            }
            break;
        }
        case InputKey::functional:
            parsed.functional = read_value();
            break;
        case InputKey::outputfile:
            parsed.outputfile = read_value();
            break;
        case InputKey::totalEConv:
            parsed.totalEConv = read_tolerance();
            break;
        case InputKey::davidsonConv:
            parsed.davidsonConv = read_tolerance();
            break;
        case InputKey::maxIter: {
            const std::string value = read_value();
            int iterations;
            std::istringstream numeric(value);
            if (!(numeric >> iterations) || iterations <= 0)
                fail("maxIter must be a positive integer");
            require_end(numeric);
            parsed.maxIter = value;
            break;
        }
        case InputKey::unknown:
            fail("Unknown entry: " + key);
            break;
        }
    }

    for (const char* required : {"latVec", "atoms", "functional", "outputfile",
                                 "totalEConv", "maxIter", "davidsonConv"})
        if (seen.count(required) == 0)
            fail(std::string("Missing entry: ") + required);

    Structure structure(lattice, coordinates, numbers, masses);
    structure.pseudo = std::move(pseudopotentials);
    cryStruct = std::move(structure);
    functional = std::move(parsed.functional);
    outputfile = std::move(parsed.outputfile);
    totalEConv = std::move(parsed.totalEConv);
    maxIter = std::move(parsed.maxIter);
    davidsonConv = std::move(parsed.davidsonConv);
}

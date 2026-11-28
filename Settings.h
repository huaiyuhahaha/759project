

#ifndef SETTINGS_H
#define SETTINGS_H
#include "structure.h"
#include <string>

class Settings {
    
public:
    Settings();
    ~Settings();
Structure cryStruct;
std::string functional;
std::string inputfile;
std::string outputfile;
std::string totalEConv;
std::string maxIter;
std::string davidsonConv;

// Read inputfile using the text format illustrated in settings.example.in.
void readInput();



};



#endif // SETTINGS_H

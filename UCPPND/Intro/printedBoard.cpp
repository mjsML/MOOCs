#include<fstream>
#include<iostream>
#include<sstream>
#include<string>
#include<vector>
using namespace std;

//strongly typed state
enum class state{Empty,Obstacle};

// Parses a single row from the board
vector<int> ParseLine(string line){
    istringstream isline(line);
    int n;
    char c;
    vector<int> row;
    while(isline>>n>>c && c==',')
    {
        row.push_back(n);
    }
    return row;
}

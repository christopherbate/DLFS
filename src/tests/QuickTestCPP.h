/**
 * QuickTestCPP.h
 * Author: Christopher Bate
 * 
 * This is a header-only framework, no need for libraries!
 * The basic idea is simply add several tests, and let the singleton take care of keeping track
 * of managing the tests. Use the assertion methods within QuickTest class within testing code.
 */
#ifndef CTB_TEST
#define CTB_TEST
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <iomanip>

/**
black        30         40
red          31         41
green        32         42
yellow       33         43
blue         34         44
magenta      35         45
cyan         36         46
white        37         47
reset             0  (everything back to normal)
bold/bright       1  (often a brighter shade of the same colour)
underline         4
inverse           7  (swap foreground and background colours)
bold/bright off  21
underline off    24
inverse off      27
**/

const std::string RED = "\033[1;31m";
const std::string GRN = "\033[32m";
const std::string RST = "\033[0m";

#define QTEqual(val1, val2) QuickTest::Equal(val1, val2, __FILE__, __LINE__)
#define QTNotEqual(val1, val2) QuickTest::NotEqual(val1, val2, __FILE__, __LINE__)
#define QTAlmostEqual(val1, val2, eps) QuickTest::AlmostEqual(val1, val2, eps, __FILE__, __LINE__)

/**
 * Test Exception class
 */
class QuickTestError : public std::exception
{
public:
    QuickTestError(const std::string &errorMsg)
        : m_errorMsg(errorMsg) {}
    const char *what()
    {
        return m_errorMsg.c_str();
    }

private:
    std::string m_errorMsg;
};

/**
 * QuickTest Assert Statements
 */
class QuickTest
{
public:
    template <typename T1, typename T2>
    static void Equal(T1 a, T2 b)
    {
        T1 c = static_cast<T1>(b);
        if (a != c)
        {
            std::stringstream ss;
            ss << "NotEqual: " << a << " != " << c;
            throw QuickTestError(ss.str());
        }
    }

    template <typename T1, typename T2>
    static void Equal(T1 a, T2 b, const char *file, const int line)
    {
        std::cout.precision(3);
        T1 c = static_cast<T1>(b);
        if (a != c)
        {
            std::stringstream ss;
            ss << "Test Failure at " << file << ":" << line
               << " NotEqual: " << a << " != " << c;
            throw QuickTestError(ss.str());
        }
    }

    static void NotEqual(void *a, void *b)
    {
        if (a == b)
        {
            std::string msg = "Equal: " + std::to_string(uint64_t(a)) + " != " + std::to_string(uint64_t(b));
            throw QuickTestError(msg);
        }
    }

    template <typename T1, typename T2>
    static void NotEqual(T1 a, T2 b, const char *file, const int line)
    {
        T1 c = static_cast<T1>(b);
        if (a == c)
        {
            std::stringstream ss;
            ss << "Test Failure at " << file << ":" << line
               << " Should not equal: " << a << " == " << c;
            throw QuickTestError(ss.str());
        }
    }

    static void AlmostEqual(float a, float b, float eps, const char *file, const int line)
    {        
        if (std::abs(a - b) > eps)
        {
            std::stringstream ss;
            ss << "Test Failure at " << file << ":" << line
               << "(AlmostEqual)" << (a-b) << " > " << eps;
            throw QuickTestError(ss.str());
        }
    }
};

/**
 * Represents an individual test.
 */
class Test
{
public:
    Test(std::string n, void (*f)())
    {
        name = n;
        func = f;
        fail = false;
    }
    void Run()
    {
        std::cout << name << ": " << std::endl;
        try
        {
            func();
            std::cout << GRN << "Test Passed" << RST << std::endl;
            fail = false;
        }
        catch (QuickTestError &e)
        {
            std::cout << RED << "Test Failed \n";
            std::cout << e.what() << RST << std::endl;
            fail = true;
        }
        catch (std::exception &e)
        {
            std::cout << RED << "Test Failed, Fatal Exception \n";
            std::cout << e.what() << RST << std::endl;
            fail = true;
        }
    }
    bool fail;
    void (*func)();
    std::string name;
};

/**
 * Represents a set of tests.
 */
class TestCase
{
public:
    TestCase(std::string name)
        : m_name(name), failCount(0) {}

    ~TestCase()
    {
        for (unsigned int i = 0; i < tests.size(); i++)
        {
            delete tests[i];
        }
    }
    void Run()
    {
        for (auto it = tests.begin(); it != tests.end(); it++)
        {
            (*it)->Run();
            std::cout << std::endl;
            if ((*it)->fail)
            {
                failCount++;
            }
        }
    }
    void PrintFailures()
    {
        for (auto it = tests.begin(); it != tests.end(); it++)
        {
            std::cout << RED;
            if ((*it)->fail)
            {
                std::cout << (*it)->name << std::endl;
            }
            std::cout << RST;
        }
    }
    std::vector<Test *> tests;
    std::string m_name;
    int failCount;
};

/**
 * TestRunner
 * Runs all tests 
 */
class TestRunner
{
public:
    /** Returns singleton instance */
    static TestRunner *GetRunner()
    {
        static TestRunner runner;
        return &runner;
    }
    ~TestRunner()
    {
        auto it = m_cases.begin();
        while (it != m_cases.end())
        {
            delete it->second;
            it++;
        }
    }

    void Run()
    {
        auto it = m_cases.begin();
        while (it != m_cases.end())
        {
            std::cout << "--------------------\n";
            std::cout << it->first << "\n";
            std::cout << "--------------------\n"
                      << std::endl;
            it->second->Run();
            it++;
        }
    }

    void RunOne(std::string test)
    {
        auto testCase = m_cases.find(test);
        if (testCase != m_cases.end())
        {
            testCase->second->Run();
        }
    }

    void PrintSummary()
    {
        std::cout << "======SUMMARY======" << std::endl;
        for (auto it = m_cases.begin(); it != m_cases.end(); it++)
        {
            std::cout << it->first << ":" << it->second->tests.size() - it->second->failCount << "/" << it->second->tests.size() << std::endl;
            it->second->PrintFailures();
        }
    }
    int GetRetCode()
    {
        return m_retCode;
    }
    void AddTest(std::string caseName, std::string testName, void (*f)())
    {
        auto testCase = m_cases.find(caseName);
        if (testCase == m_cases.end())
        {
            m_cases[caseName] = new TestCase(caseName);
        }
        m_cases[caseName]->tests.push_back(new Test(testName, f));
    }
    /** The vector of test casts (sets of tests) */
    std::unordered_map<std::string, TestCase *> m_cases;
    int m_retCode;
    TestRunner()
    {
    }
};

#endif
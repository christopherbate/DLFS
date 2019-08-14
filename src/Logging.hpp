#pragma once

#include <iostream>
#include <string>
#include <nvjpeg.h>
#include <cudnn.h>
#include <cublas.h>

const char *cudaGetErrorName(nvjpegStatus_t error);
const char *cudaGetErrorName(cudnnStatus_t error);
const char *cudaGetErrorName(cublasStatus_t error);

namespace DLFS
{

class DLFSError : public std::runtime_error
{
public:
	DLFSError(const std::string &msg) : std::runtime_error(msg)
	{
		m_errorMsg = msg;
	}
	const char *what() const throw()
	{
		return m_errorMsg.c_str();
	}

private:
	std::string m_errorMsg;
};

enum LogLevel
{
	Debug,
	Info,
	Warn,
	Error
};

class LoggingUtility
{
public:
	LoggingUtility() {}
	~LoggingUtility()
	{
		if (m_opened)
		{
			std::cout << std::endl;
		}
		m_opened = false;
	}

	template <class T>
	LoggingUtility &operator<<(const T &msg)
	{
		if (m_msgLevel >= m_minLevel)
		{
			std::cout << msg << " ";
			m_opened = true;
		}
		return *this;
	}

	void SetMinLevel(LogLevel l)
	{
		m_minLevel = l;
	}

	LoggingUtility &WARN()
	{
		m_msgLevel = Warn;
		std::cout << "\n[" << GetLabel(m_msgLevel) << "] ";
		return *this;
	}
	LoggingUtility &INFO()
	{
		m_msgLevel = Info;
		std::cout << "\n[" << GetLabel(m_msgLevel) << "] ";
		return *this;
	}
	LoggingUtility &DEBUG()
	{
		m_msgLevel = Debug;
		std::cout << "\n[" << GetLabel(m_msgLevel) << "] ";
		return *this;
	}
	LoggingUtility &ERROR()
	{
		m_msgLevel = Error;
		std::cout << "\n[" << GetLabel(m_msgLevel) << "] ";
		return *this;
	}

private:
	bool m_opened{false};

	LogLevel m_msgLevel{Warn};
	LogLevel m_minLevel{Warn};

	inline const char *GetLabel(LogLevel level)
	{
		switch (level)
		{
		case Debug:
			return "DEBUG";
		case Info:
			return "INFO";
		case Warn:
			return "WARN";
		case Error:
			return "ERROR";
		default:
			return "INFO";
		}
	}
};

extern LoggingUtility LOG;

template <typename T>
void checkWithException(T result, const char *func, const char *file,
						int const line)
{
	if (result)
	{
		std::string errorMsg =
			"CUDA error at " + std::string(file) + ":" + std::to_string(line) + " code = " +
			std::to_string(static_cast<unsigned int>(result)) +
			" " + cudaGetErrorName(result) + " " + std::string(func);
		throw DLFS::DLFSError(errorMsg);
	}
}

#ifdef __DRIVER_TYPES_H__
#define checkCudaErrors(val) checkWithException((val), #val, __FILE__, __LINE__)
#endif

} // namespace DLFS

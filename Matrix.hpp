#ifndef EX3_MATRIX_HPP
#define EX3_MATRIX_HPP

/***************************************************************************************************
 * Includes.
 **************************************************************************************************/
#include <iostream>
#include <vector>
#include <stdexcept> //for exception, runtime_error, out_of_range.
#include <sstream>
#include <thread>
#include <mutex>
#include "Complex.h"

/***************************************************************************************************
 * Matrix Header File.
 **************************************************************************************************/
/**
 * Class for a generic Matrix data type.
 */
template <class T>
class Matrix
{
public:

	//---------------- Typedefs ----------------//
	typedef typename std::vector<T>::const_iterator const_iterator;

	//---------------- Constructors ----------------//
	/**
	 * Default Constructor.
	 * Creates a matrix of dimension 1X1 containing 0.
	 * @Throws: bad_alloc if inserting new element to the vector fails.
	 */
	Matrix();

	/**
	 * A Constructor.
	 * Creates a matrix of dimension row X col containing 0 in all cells.
	 * @param: rows number of rows.
	 * @param: cols number of columns.
	 * @Throws: bad_alloc if inserting new element to the vector fails.
	 */
	Matrix(unsigned int rows, unsigned int cols);

	/**
	 * Copy Constructor.
	 * Uses stl::vector assignment operator.
	 * @Throws: Vector's operator = Exception safety.
	 *          Basic guarantee: if an exception is thrown, the container is in a valid state.
	 */
	Matrix(const Matrix& otherMat);

	/**
	 * Move Constructor.
	 * Uses stl::vector assignment operator.
	 * @Throws: Vector's operator=: Exception safety.
	 * Basic guarantee: if an exception is thrown, the container is in a valid state.
	 */
	Matrix(Matrix && otherMat);

	/**
	 * A Constructor.
	 * Creates a matrix of dimension row X col containing the elements from the given vector.
	 * Order of the elements is determent by the Matrix's Iterator.
	 * We assume T type has a Copy Constructor.
	 * @param: rows number of rows.
	 * @param: cols number of columns.
	 * @param: cells vector of elements to insert the matrix.
	 * @Throws: bad_alloc if inserting new element to the vector fails.
	 * @Throws: length_error if the number of elements to insert the matrix, is different from the
	 *          matrix's requested size.
	 */
	Matrix(unsigned int rows, unsigned int cols, const std::vector<T>& cells) throw();

	//---------------- Destructor ----------------//
	/**
	 * The Destructor.
	 */
	~Matrix();

	//---------------- Operators Overloading ----------------//
	/*
	 * Assigns the value of the matrix on the right of the expression to the one on the left.
	 * It creates a new, identical and independent copy of the right matrix.
	 * @Param: otherMat - Matrix<T> type.
	 */
	Matrix& operator = (const Matrix &otherMat);

	/**
	 * Adds up two matrices.
	 * The compiler will convert the returned object to rvalue when needed.
	 * @Param: otherMat - Matrix<T> type.
	 * @Throws: length_error if trying to add matrices of different sizes.
	 */
	Matrix<T> operator + (const Matrix<T>& otherMat) const throw();

	/**
	 * Subtracts one matrix from the other.
	 * The compiler will convert the returned object to rvalue when needed.
	 * @Param: otherMat - Matrix<T> type.
	 * @Throws: length_error if trying to subtract matrices of different sizes.
	 */
	Matrix<T> operator - (const Matrix<T>& otherMat) const throw();

	/**
	 * Multiplies one matrix with the other.
	 * Multiplication is posible for matrices of sizes: n x m, m x p.
	 * The compiler will convert the returned object to rvalue when needed.
	 * @Param: otherMat - Matrix<T> type.
	 * @Throws: length_error if trying to multiply a matrix with number of columns different from
	 * the other matrix's number of rows.
	 */
	Matrix<T> operator * (const Matrix<T>& otherMat) const throw();

	/**
	 * @Param: otherMat - Matrix<T> type.
	 * @Returns True if the two matrices have the same T values at the same (row, col) index in
	 * each matrix.
	 */
	bool operator == (const Matrix<T>& otherMat) const;

	/**
	 * @Param: otherMat - Matrix<T> type.
	 * @Returns True if the two matrices have at least one T value that is different at the same
	 * (row, col) index in each matrix.
	 */
	bool operator != (const Matrix<T>& otherMat) const;

	/**
	 * Prints the matrix using ostream object.
	 * Each row of the matrix is printed in a separated line
	 * and TAB character separates between each cell's value.
	 * @Param: otherMat - Matrix<T> type.
	 * @Param: currentStream - the current output stream.
	 * @Returns the updated stream.
	 */
	template<class U>
	friend std::ostream& operator << (std::ostream& currentStream, const Matrix<U>& mat);

	/**
	 * @Returns a copy of the requested cell (mat[row,col]) value.
	 * @param: rows number of rows.
	 * @param: cols number of columns.
	 * @Throws: out_of_range if requesting an element out of the matrix's boundaries.
	 */
	const T& operator () (const unsigned int row, const unsigned int col) const throw();

	/**
	 * @Returns a reference to the requested cell (mat[row,col]).
	 * Allows changing of the cell's value with the returned type.
	 * @param: rows number of rows.
	 * @param: cols number of columns.
	 * @Throws: out_of_range if requesting an element out of the matrix's boundaries.
	 */
	T& operator () (const unsigned int row, const unsigned int col) throw();

	//---------------- Methods ----------------//
	/**
	 * @Returns a new object of the current matrix transposed.
	 * The is also a specialized instance of the 'trans()' function for the given Complex class.
	 */
	Matrix<T> trans();

	/**
	 * @Returns True iff the current matrix is squared (Number of rows equals number of columns).
	 */
	bool isSquareMatrix() const;

	/**
	 * Switch variable content of one matrix with the content of the variable of the other matrix,
	 * For all of Matrix class variables.
	 * @Param: mat1, mat2 - Matrix<T> type.
	 */
	template<class U>
	friend void swap(Matrix<U> &mat1, Matrix<U> &mat2);

	/**
	 * @Returns the number of rows of the matrix.
	 */
	unsigned int rows() const;

	/**
	 * @Returns the number of columns of the matrix.
	 */
	unsigned int cols() const;

	/**
	 * If state is True, then activates parallel calculation of the operators '+', '*' using
	 * multiple threads.
	 * Else, if state is False, calculate these operators results using a single thread.
	 * @Param state is a Boolean used to determine the calculation method as stated above.
	 */
	static void setParallel(bool state);

	//---------------- Iterators ----------------//
	/**
	 * @Returns an iterator to the first cell of the matrix.
	 * Iterates by the order: '‫‪(0,0)­>(0,1)­>...­>(0,col­1)­>(1,0)­>....­>(row­1,col­1)'.
	 */
	typename std::vector<T>::const_iterator begin();

	/**
	 * @Returns an iterator to the last cell of the matrix.
	 */
	typename std::vector<T>::const_iterator end();

private:
	//---------------- Variables ----------------//
	/**
	 * Uses a single vector to represent the Matrix data structure.
	 * Accessing the matrix at matrix(i,j) when implemented by the vector: vector[(col * i) + j].
	 */
	std::vector<T> _matVec;

	/**
	 * The number of the matrix's rows, non-negative number only.
	 */
	unsigned int _rows;

	/**
	 * The number of the matrix's cols, non-negative number only.
	 */
	unsigned int _cols;

	/**
	 * Save the current state of the multi-threading mode.
	 */
	static bool _parallelMode;

	//---------------- Constants ----------------//
	/**
	 * Number of rows for the default constructor.
	 */
	static const unsigned int s_defaultRowSize;

	/**
	 * Number of columns for the default constructor.
	 */
	static const unsigned int s_defaultColSize;

	/**
	 * TAB Character.
	 */
	static const char* s_tab;

	/**
	 * Exception Message:
	 * Number of elements to insert the matrix, is different from the matrix's requested size.
	 */
	static const char* s_vecMatInitBadSizeExcMsg;

	/**
	 * Exception Message:
	 * Adding matrices of different sizes is not defined.
	 */
	static const char* s_addMatSizeExcMsg;

	/**
	 * Exception Message:
	 * Subtracting matrices of different sizes is not defined.
	 */
	static const char* s_subMatSizeExcMsg;

	/**
	 * Exception Message:
	 * Multiplying a matrix with number of columns different from the other matrix's number of
	 * rows is not defined.
	 */
	static const char* s_mulMatSizeExcMsg;

	/**
	 * Exception Message:
	 * Requesting an element out of the matrix's boundaries.
	 */
	static const char* s_matOutOfBoundsExcMsg;

	/**
	 * Message printed upon changing state to multi-threaded processing.
	 */
	static const char* s_parallelMsg;

	/**
	 * Message printed upon changing state to single-threaded processing.
	 */
	static const char* s_nonParallelMsg;

	//---------------- Methods ----------------//
	/**
	 * The logic used for adding matrices.
	 * Used for one or more threads.
	 */
	static void _addLogic(Matrix<T> &newMat, const Matrix<T> &otherMat, unsigned int begIndex,
				          unsigned int endIndex);

	/**
	 * The logic used for multiplying matrices.
	 * Used for one or more threads.
	 */
	static void _mulLogic(Matrix<T> &newMat, const Matrix<T> &leftMat, const Matrix<T> &rightMat,
						  unsigned int startRow, unsigned int endRow);
};

/***************************************************************************************************
 * Constants.
 **************************************************************************************************/
template <class T>
const unsigned int Matrix<T>::s_defaultRowSize = 1;

template <class T>
const unsigned int Matrix<T>::s_defaultColSize = 1;

template <class T>
const char* Matrix<T>::s_tab = "\t";

template <class T>
const char* Matrix<T>::s_vecMatInitBadSizeExcMsg = "Number of elements to insert the matrix, is "
		                                           "different from the matrix's requested size.";

template <class T>
const char* Matrix<T>::s_addMatSizeExcMsg = "Adding matrices of different sizes is not defined.";

template <class T>
const char* Matrix<T>::s_subMatSizeExcMsg = "Subtracting matrices of different sizes is not "
		                                    "defined.";

template <class T>
const char* Matrix<T>::s_mulMatSizeExcMsg = "Multiplying a matrix with number of columns different "
	                                   	    "from the other matrix's number of rows is not "
		                                    "defined.";
template <class T>
const char* Matrix<T>::s_matOutOfBoundsExcMsg = "Requesting an element out of the matrix's "
		                                        "boundaries.";

template <class T>
const char* Matrix<T>::s_parallelMsg = "Generic Matrix mode changed to Parallel mode.";

template <class T>
const char* Matrix<T>::s_nonParallelMsg = "Generic Matrix mode changed to non-Parallel mode.";

/***************************************************************************************************
 * Static Variables Init.
 **************************************************************************************************/
template <class T>
bool Matrix<T>::_parallelMode = false;

/***************************************************************************************************
 * An implementation of the generic (template) matrix class.
 **************************************************************************************************/
/**
 * Default Constructor.
 * Creates a matrix of dimension 1X1 containing 0.
 * @Throws: bad_alloc if inserting new element to the vector fails.
 */
template<class T>
Matrix<T>::Matrix() : _rows(s_defaultRowSize), _cols(s_defaultColSize)
{
	_matVec.push_back(T(0)); //It is assumed T(0) exists.
}

/**
 * A Constructor.
 * Creates a matrix of dimension row X col containing 0 in all cells.
 * @param: rows number of rows.
 * @param: cols number of columns.
 * @Throws: bad_alloc if inserting new element to the vector fails.
 */
template<class T>
Matrix<T>::Matrix(unsigned int rows, unsigned int cols) : _rows(rows), _cols(cols)
{
	for(unsigned int i = 0; i < rows * cols; ++i)
	{
		_matVec.push_back(T(0)); //It is assumed T(0) exists.
	}
}

/**
 * Copy Constructor.
 * Uses stl::vector assignment operator.
 * @Throws: Vector's operator=: Exception safety.
 *          Basic guarantee: if an exception is thrown, the container is in a valid state.
 */
template<class T>
Matrix<T>::Matrix(const Matrix& otherMat) : _matVec(otherMat._matVec), _rows(otherMat._rows),
											_cols(otherMat._cols) {}

/**
 * Move Constructor.
 * Uses stl::vector assignment operator.
 * @Throws: Vector's operator = Exception safety.
 * Basic guarantee: if an exception is thrown, the container is in a valid state.
 */
template<class T>
Matrix<T>::Matrix(Matrix && otherMat) : _matVec(std::move(otherMat._matVec)), _rows(otherMat._rows),
									    _cols(otherMat._cols) {}

/**
 * A Constructor.
 * Creates a matrix of dimension row X col containing the elements from the given vector.
 * Order of the elements is determent by the Matrix's Iterator.
 * We assume T type has a Copy Constructor.
 * @param: rows number of rows.
 * @param: cols number of columns.
 * @param: cells vector of elements to insert the matrix.
 * @Throws: bad_alloc if inserting new element to the vector fails.
 * @Throws: length_error if the number of elements to insert the matrix, is different from the
 *          matrix's requested size.
 */
template<class T>
Matrix<T>::Matrix(unsigned int rows, unsigned int cols, const std::vector<T>& cells) throw() :
		          _rows(rows), _cols(cols)
{
	if((rows * cols) != cells.size())
	{
		throw std::length_error(s_vecMatInitBadSizeExcMsg);
	}

	typename std::vector<T>::const_iterator it = cells.begin();
	for(; it != cells.end(); it++)
	{
		this -> _matVec.push_back((*it));
	}
}

//---------------- Destructor ----------------//
/**
 * The Destructor.
 */
template<class T>
Matrix<T>::~Matrix() {}

//---------------- Operators Overloading ----------------//
/**
 * Assigns the value of the matrix on the right of the expression to the one on the left.
 * It creates a new, identical and independent copy of the right matrix.
 * @Param: otherMat - Matrix<T> type.
 */
template<class T>
Matrix<T>& Matrix<T>::operator = (const Matrix<T> &otherMat)
{
	if(this != &otherMat)
	{
		Matrix<T> tmpMat = otherMat;
		swap(*this, tmpMat); //Never throws exceptions.
	}
	return *this;
}

/**
 * Adds up two matrices.
 * The compiler will convert the returned object to rvalue when needed.
 * @Param: otherMat - Matrix<T> type.
 * @Throws: length_error if trying to add matrices of different sizes.
 */
template<class T>
Matrix<T> Matrix<T>::operator + (const Matrix<T>& otherMat) const throw()
{
	//Exceptions check.
	if((_matVec.size() != otherMat._matVec.size()) || (_rows != otherMat._rows) ||
	   (_cols != otherMat._cols))
	{
		throw std::length_error(s_addMatSizeExcMsg);
	}

	Matrix<T> newMat(*this); //New matrix to be returned.
	if(!_parallelMode)
	{
		//Start and end position is the entire matrix's vector.
		_addLogic(newMat, otherMat, 0, _rows * _cols);
	}
	else
	{
		std::vector<std::thread> threadPerLineVec; //New vector for keeping created threads.
		for(unsigned int threadNum = 0; threadNum < _rows; ++threadNum)
		{
			//Each thread gets the beginging and end of the next line to calculate.
			threadPerLineVec.push_back(std::thread(&Matrix<T>::_addLogic, std::ref(newMat),
												   otherMat,
												   threadNum * _cols,
												   (threadNum * _cols) + _cols));
		}
		for(std::thread &thread : threadPerLineVec) //Join threads.
		{
			thread.join();
		}
	}
	return newMat;
}

/**
 * Subtracts one matrix from the other.
 * The compiler will convert the returned object to rvalue when needed.
 * @Param: otherMat - Matrix<T> type.
 * @Throws: length_error if trying to subtract matrices of different sizes.
 */
template<class T>
Matrix<T> Matrix<T>::operator - (const Matrix<T>& otherMat) const throw()
{
	if((_matVec.size() != otherMat._matVec.size()) || (_rows != otherMat._rows) ||
	   (_cols != otherMat._cols))
	{
		throw std::length_error(s_subMatSizeExcMsg);
	}

	Matrix<T> newMat(*this);
	for(unsigned int i = 0; i < newMat._matVec.size(); ++i)
	{
		newMat._matVec[i] -= otherMat._matVec[i]; //It is assumed -= of T exists.
	}
	return newMat;
}

/**
 * Multiplies one matrix with the other.
 * Multiplication is posible for matrices of sizes: n x m, m x p.
 * The compiler will convert the returned object to rvalue when needed.
 * @Param: otherMat - Matrix<T> type.
 * @Throws: length_error if trying to multiply a matrix with number of columns different from
 * the other matrix's number of rows.
 */
template<class T>
Matrix<T> Matrix<T>::operator * (const Matrix<T>& otherMat) const throw()
{
	//Exceptions check.
	unsigned int n = _rows, m = _cols, p = otherMat._cols;
	if(m != (otherMat._rows)) //Is 'otherMat._rows' is also |m| ?
	{
		throw std::length_error(s_mulMatSizeExcMsg);
	}

	//Variables.
	Matrix<T> newMat(n, p); //New matrix to be returned (n x p).

	if(!_parallelMode)
	{
		_mulLogic(newMat, (*this), otherMat, 0, n); //First to last row of the matrix.
	}
	else
	{
		std::vector<std::thread> threadPerLineVec; //New vector for keeping created threads.
		for(unsigned int threadNum = 0; threadNum < n; ++threadNum)
		{
			//Each thread handles one line.
			threadPerLineVec.push_back(std::thread(&Matrix<T>::_mulLogic, std::ref(newMat),
												   std::ref(*this), otherMat, threadNum,
												   threadNum + 1));
		}
		for(std::thread &thread : threadPerLineVec) //Join threads.
		{
			thread.join();
		}
	}
	return newMat;
}

/**
 * @Param: otherMat - Matrix<T> type.
 * @Returns True if the two matrices have the same T values at the same (row, col) index in
 * each matrix.
 */
template<class T>
bool Matrix<T>::operator == (const Matrix<T>& otherMat) const
{
	return ((this == &otherMat) || ((_matVec == otherMat._matVec) && (_rows == otherMat._rows) &&
			                        (_cols == otherMat._cols)));
}

/**
 * @Param: otherMat - Matrix<T> type.
 * @Returns True if the two matrices have at least one T value that is different at the same
 * (row, col) index in each matrix.
 */
template<class T>
bool Matrix<T>::operator != (const Matrix<T>& otherMat) const
{
	return !(*this == otherMat);
}

/**
 * Prints the matrix using ostream object.
 * Each row of the matrix is printed in a separated line
 * and TAB character separates between each cell's value.
 * @Param: otherMat - Matrix<T> type.
 * @Param: currentStream - the current output stream.
 * @Returns the updated stream.
 */
template<class T>
std::ostream& operator << (std::ostream& currentStream, const Matrix<T>& mat)
{
	if(mat._matVec.size() > 0)
	{
		currentStream << mat._matVec[0] << Matrix<T>::s_tab;
		for(unsigned int i = 1; i < mat._matVec.size(); ++i)
		{
			if((i % mat._cols) != 0)
			{
				currentStream << mat._matVec[i] << Matrix<T>::s_tab;
			}
			else
			{
				currentStream << std::endl << mat._matVec[i] << Matrix<T>::s_tab;
			}
		}
	}
	currentStream << std::endl;
	return currentStream;
}

/**
 * @Returns a copy of the requested cell (mat[row,col]) value.
 * @param: rows number of rows.
 * @param: cols number of columns.
 * @Throws: out_of_range if requesting an element out of the matrix's boundaries.
 */
template<class T>
const T& Matrix<T>::operator () (const unsigned int row, const unsigned int col) const throw()
{
	if((row >= _rows) || (col >= _cols))
	{
		throw std::out_of_range(s_matOutOfBoundsExcMsg);
	}
	return _matVec[row * _cols + col];
}

/**
 * @Returns a reference to the requested cell (mat[row,col]).
 * Allows changing of the cell's value with the returned type.
 * @param: rows number of rows.
 * @param: cols number of columns.
 * @Throws: out_of_range if requesting an element out of the matrix's boundaries.
 */
template<class T>
T& Matrix<T>::operator () (const unsigned int row, const unsigned int col) throw()
{
	if((row >= _rows) || (col >= _cols))
	{
		throw std::out_of_range(s_matOutOfBoundsExcMsg);
	}
	return _matVec[row * _cols + col];
}

//---------------- Methods ----------------//
/**
 * @Returns a new object of the current matrix transposed.
 */
template<class T>
Matrix<T> Matrix<T>::trans()
{
	Matrix<T> newMat(_cols, _rows);
	for(unsigned int i = 0; i < _rows; ++i)
	{
		for(unsigned int j = 0; j < _cols; ++j)
		{
			newMat(j, i) = operator()(i, j);
		}
	}
	return newMat;
}

/**
 * Specialization of the 'trans()' function for the given Complex class.
 * @Returns a new object of the current matrix Conjugated.
 */
template<>
Matrix<Complex> Matrix<Complex>::trans()
{
	Matrix<Complex> newMat(_cols, _rows);
	for(unsigned int i = 0; i < _rows; ++i)
	{
		for(unsigned int j = 0; j < _cols; ++j)
		{
			newMat(j, i) = (operator()(i, j)).conj();
		}
	}
	return newMat;
}

/**
 * @Returns True iff the current matrix is squared (Number of rows equals number of columns).
 */
template<class T>
bool Matrix<T>::isSquareMatrix() const
{
	return (_rows == _cols);
}

/**
 * Switch variable content of one matrix with the content of the variable of the other matrix,
 * For all of Matrix class variables.
 * @Param: mat1, mat2 - Matrix<T> type.
 */
template<class T>
void swap(Matrix<T> &mat1, Matrix<T> &mat2)
{
	//All built-in swaps throw no exceptions.
	std::swap(mat1._matVec, mat2._matVec);
	std::swap(mat1._rows, mat2._rows);
	std::swap(mat1._cols, mat2._cols);
}

/**
 * @Returns the number of rows of the matrix.
 */
template<class T>
unsigned int Matrix<T>::rows() const
{
	return _rows;
}

/**
 * @Returns the number of columns of the matrix.
 */
template<class T>
unsigned int Matrix<T>::cols() const
{
	return _cols;
}

/**
 * If state is True, then activates parallel calculation of the operators '+', '*' using
 * multiple threads.
 * Else, if state is False, calculate these operators results using a single thread.
 * @Param state is a Boolean used to determine the calculation method as stated above.
 */
template<class T>
void Matrix<T>::setParallel(bool state)
{
	if(_parallelMode != state)
	{
		if(state)
		{
			_parallelMode = true;
			std::cout << s_parallelMsg << std::endl;
		}
		else
		{
			_parallelMode = false;
			std::cout << s_nonParallelMsg << std::endl;
		}
	}
}

/**
 * The logic used for adding matrices.
 * Used for one or more threads.
 */
template<class T>
void Matrix<T>::_addLogic(Matrix<T> &newMat, const Matrix<T> &otherMat, unsigned int begIndex,
			              unsigned int endIndex)
{
	for(unsigned int i = begIndex; i < endIndex; ++i)
	{
		newMat._matVec[i] += otherMat._matVec[i]; //It is assumed += of T exists.
	}
}

/**
 * The logic used for multiplying matrices.
 * Used for one or more threads.
 */
template<class T>
void Matrix<T>::_mulLogic(Matrix<T> &newMat, const Matrix<T> &leftMat, const Matrix<T> &rightMat,
						  unsigned int startRow, unsigned int endRow)
{
	unsigned int m = leftMat._cols, p = rightMat._cols;
	T tmpSum(0); //It is assumed T(0) exists.
	std::mutex mulMutex;

	for(unsigned int i = startRow; i < endRow; ++i) //Multiplication iterative algorithm.
	{
		for(unsigned int j = 0; j < p; ++j)
		{
			std::lock_guard<std::mutex> lock(mulMutex);
			tmpSum = 0;  //It is assumed '=' of T exists.
			for(unsigned int k = 0; k < m; ++k)
			{
				tmpSum += leftMat(i, k) * rightMat(k, j);  //It is assumed '+=' of T exists.
			}
			newMat(i, j) = tmpSum;
		}
	}
}

//---------------- Iterators ----------------//
/**
 * @Returns an iterator to the first cell of the matrix.
 * Iterates by the order: '‫‪(0,0)­>(0,1)­>...­>(0,col­1)­>(1,0)­>....­>(row­1,col­1)'.
 */
template<class T>
typename Matrix<T>::const_iterator Matrix<T>::begin()
{
	return _matVec.begin();
}

/**
 * @Returns an iterator to the last cell of the matrix.
 */
template<class T>
typename Matrix<T>::const_iterator Matrix<T>::end()
{
	return _matVec.end();
}
#endif //EX3_MATRIX_HPP
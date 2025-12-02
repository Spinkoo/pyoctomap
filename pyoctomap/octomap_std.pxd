# Standard library type declarations for OctoMap wrapper

from libcpp cimport bool
from libcpp.string cimport string
from libc.stddef cimport size_t

cdef extern from * nogil:
    cdef T dynamic_cast[T](void *) except +   # nullptr may also indicate failure
    cdef T static_cast[T](void *)
    cdef T reinterpret_cast[T](void *)
    cdef T const_cast[T](void *)

cdef extern from "<iostream>" namespace "std":
    cdef cppclass istream:
        istream() except +
    cdef cppclass ostream:
        ostream() except +

cdef extern from "<fstream>" namespace "std":
    cdef cppclass ifstream(istream):
        ifstream() except +
        ifstream(const char*) except +
        bint is_open()
        void open(const char*)
        void close()
    cdef cppclass ofstream(ostream):
        ofstream() except +
        ofstream(const char*) except +
        bint is_open()
        void open(const char*)
        void close()

cdef extern from "<sstream>" namespace "std":
    cdef cppclass istringstream:
        istringstream() except +
        istringstream(string& s) except +
        string str()
        void str(string& s)
    cdef cppclass ostringstream:
        ostringstream() except +
        string str()

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        cppclass iterator:
            T& operator*()
            iterator& operator++()
            bool operator==(iterator& other)
            bool operator!=(iterator& other)
        cppclass const_iterator:
            const T& operator*()
            const_iterator& operator++()
            bool operator==(const_iterator& other)
            bool operator!=(const_iterator& other)
        vector()
        size_t size()
        T& operator[](size_t)
        const T& operator[](size_t) const
        T& back()
        const T& back() const
        void push_back(T&)
        void clear()
        void reserve(size_t)
        iterator begin()
        iterator end()
        const_iterator begin() const
        const_iterator end() const

cdef extern from "<list>" namespace "std":
    cdef cppclass list[T]:
        void push_back(T&)
        size_t size()
        T& front()
        T& back()
        void pop_front()
        void pop_back()
        void clear()
        cppclass iterator:
            T& operator*()
            iterator& operator++()
            iterator operator--()
            bint operator==(iterator)
            bint operator!=(iterator)
        iterator begin()
        iterator end()


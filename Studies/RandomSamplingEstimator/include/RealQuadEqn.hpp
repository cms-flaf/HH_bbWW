#ifndef REAL_QUAD_EQN_HPP
#define REAL_QUAD_EQN_HPP

#include <type_traits>
#include <stdexcept>
#include <vector>

// ax^2 + bx + c = 0
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
class QuadEqn 
{
    public:
    QuadEqn(T a, T b, T c, double tol = 10e-9) : m_solutions(), m_a(a), m_b(b), m_c(c), m_discriminant(b*b - 4*a*c), m_tol(tol) 
    {
        if (Equal(m_a, 0.0) && Equal(m_b, 0.0) && !Equal(m_c, 0.0))
        {
            throw std::invalid_argument("Attempted to construct inconsistent equation");
        }
        m_solutions.reserve(2);
        Solve(); 
    } 

    inline std::vector<double> const& Solutions() const& { return m_solutions; }
    inline std::vector<double>&& Solutions() && { return std::move(m_solutions); }

    void Solve() 
    {
        if (Equal(m_a, 0.0))
        {
            if (!Equal(m_b, 0.0))
            {
                m_solutions.push_back(-m_c/m_b);
            }
        }
        else
        {
            if (Equal(m_discriminant, 0.0))
            {
                double x = -m_b/(2.0*m_a);
                m_solutions.push_back(x);
                m_solutions.push_back(x);
            }
            else
            {
                m_solutions.push_back((-m_b + std::sqrt(m_discriminant))/(2.0*m_a));
                m_solutions.push_back((-m_b - std::sqrt(m_discriminant))/(2.0*m_a));
            }
        }
    }

    private:
    std::vector<double> m_solutions;
    T m_a;
    T m_b;
    T m_c;
    double m_discriminant;
    double m_tol;

    inline bool Equal(double x, double y) { return std::abs(x - y) <= m_tol * std::abs(x); }
};

#endif


#include "gmm/gmm_kernel.h"

using std::endl; using std::cout; using std::cerr;
using std::ends; using std::cin;
using gmm::size_type;
bool print_debug = false;

template <typename MAT1, typename VECT1>
bool test_procedure(const MAT1 &m1_, const VECT1 &v1_) {
  VECT1 &v1 = const_cast<VECT1 &>(v1_);
  MAT1  &m1 = const_cast<MAT1  &>(m1_);
  typedef typename gmm::linalg_traits<MAT1>::value_type T;
  typedef typename gmm::number_traits<T>::magnitude_type R;
  R prec = gmm::default_tol(R());

  size_type m = gmm::mat_nrows(m1);

  R norm = gmm::vect_norm2_sqr(v1);

  R normtest(0);

  for (size_type i = 0; i < m; ++i) {
    T x(1), y = v1[i];;
    x *= v1[i];
    x += v1[i];
    x += v1[i];
    x -= v1[i];
    x -= y;
    x *= v1[i];
    x /= v1[i];
    GMM_ASSERT1(y == v1[i], "Error in basic operations");
    normtest += gmm::abs_sqr(x);
  }
  
  GMM_ASSERT1(gmm::abs(norm - normtest) <= prec * R(100),
	      "Error in basic operations");
  
  return true;
}



int main(void) {

  srand(9634);

  gmm::set_warning_level(1);

  for (int iter = 0; iter < 100000; ++iter) {

    try {

      gmm::col_matrix<gmm::wsvector<float> >  param0(2, 2);
      gmm::fill_random(param0, 0.2);
      std::vector<float>  param1(2);
      gmm::fill_random(param1);
    

      bool ret = test_procedure(param0, param1);
      if (ret) return 0;

    }
    GMM_STANDARD_CATCH_ERROR;
  }
  return 0;
}

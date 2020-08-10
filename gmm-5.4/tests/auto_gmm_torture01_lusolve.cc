

#include "gmm/gmm_kernel.h"
#include "gmm/gmm_dense_lu.h"
#include "gmm/gmm_condition_number.h"

using std::endl; using std::cout; using std::cerr;
using std::ends; using std::cin;
using gmm::size_type;
bool print_debug = false;

template <typename MAT1, typename VECT1, typename VECT2>
bool test_procedure(const MAT1 &m1_, const VECT1 &v1_, const VECT2 &v2_) {
  VECT1 &v1 = const_cast<VECT1 &>(v1_);
  VECT2 &v2 = const_cast<VECT2 &>(v2_);
  MAT1  &m1 = const_cast<MAT1  &>(m1_);
  typedef typename gmm::linalg_traits<MAT1>::value_type T;
  typedef typename gmm::number_traits<T>::magnitude_type R;
  R prec = gmm::default_tol(R());
  static size_type nb_iter = 0;

  size_type m = gmm::mat_nrows(m1);
  std::vector<T> v3(m);

  R det = gmm::abs(gmm::lu_det(m1)), error;
  R cond = gmm::condition_number(m1);

  if (print_debug) cout << "cond = " << cond << " det = " << det << endl;
  GMM_ASSERT1(det != R(0) || cond >= R(0.01) / prec || cond == R(0),
	      "Inconsistent condition number: " << cond);

  if (prec * cond < R(1)/R(10000) && det != R(0)) {
    ++nb_iter;

    gmm::lu_solve(m1, v1, v2);
    gmm::mult(m1, v1, gmm::scaled(v2, T(-1)), v3);

    error = gmm::vect_norm2(v3);
    GMM_ASSERT1(error <= prec * cond * R(20000), "Error too large: " << error);

    gmm::lu_inverse(m1);
    gmm::mult(m1, v2, v1);
    gmm::lu_inverse(m1);
    gmm::mult(m1, v1, gmm::scaled(v2, T(-1)), v3);
    
    error = gmm::vect_norm2(v3);
    GMM_ASSERT1(error <= prec * cond * R(20000), "Error too large: " << error);

    if (nb_iter == 100) return true;
  }
  return false;
}



int main(void) {

  srand(8098);

  gmm::set_warning_level(1);

  for (int iter = 0; iter < 100000; ++iter) {

    try {

      gmm::row_matrix<gmm::wsvector<std::complex<double> > >  param0(69, 28);
      gmm::size_type param_t0 [28]= {2 , 4 , 5 , 10 , 12 , 17 , 21 , 25 , 26 , 27 , 28 , 29 , 30 , 33 , 34 , 35 , 36 , 37 , 38 , 42 , 46 , 47 , 55 , 56 , 57 , 59 , 61 , 62};
      gmm::fill_random(param0);
      std::vector<std::complex<double> >  param1(28);
      gmm::fill_random(param1);
      gmm::rsvector<std::complex<double> >  param2(28);
      gmm::fill_random(param2);
    

      bool ret = test_procedure(gmm::sub_matrix(param0, gmm::sub_index(&param_t0 [0], &param_t0 [28]), gmm::sub_interval(0, 28)), param1, param2);
      if (ret) return 0;

    }
    GMM_STANDARD_CATCH_ERROR;
  }
  return 0;
}

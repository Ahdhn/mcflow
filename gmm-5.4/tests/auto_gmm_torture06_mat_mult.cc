

#include "gmm/gmm_kernel.h"

using std::endl; using std::cout; using std::cerr;
using std::ends; using std::cin;
using gmm::size_type;

template <typename MAT1, typename MAT2, typename MAT3>
bool test_procedure(const MAT1 &m1_, const MAT2 &m2_, const MAT3 &m3_) {
  MAT1  &m1 = const_cast<MAT1  &>(m1_);
  MAT2  &m2 = const_cast<MAT2  &>(m2_);
  MAT3  &m3 = const_cast<MAT3  &>(m3_);
  typedef typename gmm::linalg_traits<MAT1>::value_type T;
  typedef typename gmm::number_traits<T>::magnitude_type R;
  R prec = gmm::default_tol(R());
  static size_type nb_iter(0);
  ++nb_iter;

  size_type k = gmm::mat_nrows(m1);
  size_type l = std::max(gmm::mat_ncols(m1), gmm::mat_nrows(m2));
  size_type m = std::max(gmm::mat_ncols(m2), gmm::mat_nrows(m3));
  size_type n = gmm::mat_ncols(m3);

  gmm::dense_matrix<T> m4(k, m);
  gmm::mult(m1, m2, m4);

  R error = mat_euclidean_norm(m4)
    - mat_euclidean_norm(m1) * mat_euclidean_norm(m2);
  if (error > prec * R(100))
    GMM_ASSERT1(false, "Inconsistence of Frobenius norm" << error);

  error = mat_norm1(m4) - mat_norm1(m1) * mat_norm1(m2);
  if (error > prec * R(100))
    GMM_ASSERT1(false, "Inconsistence of norm1 for matrices"
	      << error);
  error = mat_norminf(m4) - mat_norminf(m1) * mat_norminf(m2);
  if (error > prec * R(100))
    GMM_ASSERT1(false, "Inconsistence of norminf for matrices"
		<< error);

  size_type mm = std::min(m, k);
  size_type nn = std::min(n, m);

  gmm::dense_matrix<T> m1bis(mm, l), m2bis(l, nn), m3bis(mm, nn);
  gmm::copy(gmm::sub_matrix(m1, gmm::sub_interval(0,mm),
			    gmm::sub_interval(0,l)), m1bis);
  gmm::copy(gmm::sub_matrix(m2, gmm::sub_interval(0,l),
			    gmm::sub_interval(0,nn)), m2bis);
  gmm::mult(m1bis, m2bis, m3bis);
  gmm::mult(gmm::sub_matrix(m1, gmm::sub_interval(0,mm),
			    gmm::sub_interval(0,l)),
	    gmm::sub_matrix(m2, gmm::sub_interval(0,l),
			    gmm::sub_interval(0,nn)),
	    gmm::sub_matrix(m3, gmm::sub_interval(0,mm),
			    gmm::sub_interval(0,nn)));
  gmm::add(gmm::scaled(m3bis, T(-1)),
	   gmm::sub_matrix(m3, gmm::sub_interval(0,mm),
			   gmm::sub_interval(0,nn)));
  
  error = gmm::mat_euclidean_norm(gmm::sub_matrix(m3, gmm::sub_interval(0,mm),
					   gmm::sub_interval(0,nn)));

  if (!(error <= prec * R(10000)))
    GMM_ASSERT1(false, "Error too large: " << error);

  if (nn <= gmm::mat_nrows(m3) && mm <= gmm::mat_ncols(m3)) {
    
    gmm::scale(m1, T(2));
    gmm::mult(gmm::scaled(gmm::sub_matrix(m1, gmm::sub_interval(0,mm),
					  gmm::sub_interval(0,l)), T(-1)),
	      gmm::sub_matrix(m2, gmm::sub_interval(0,l),
			      gmm::sub_interval(0,nn)),
	      gmm::sub_matrix(gmm::transposed(m3), gmm::sub_interval(0,mm),
			      gmm::sub_interval(0,nn)));
    gmm::add(gmm::scaled(m3bis, T(2)),
	     gmm::transposed(gmm::sub_matrix(m3, gmm::sub_interval(0,nn),
					     gmm::sub_interval(0,mm))));
    
    error = gmm::mat_euclidean_norm(gmm::sub_matrix(m3, gmm::sub_interval(0,nn),
					   gmm::sub_interval(0,mm)));
    
    if (!(error <= prec * R(10000)))
      GMM_ASSERT1(false, "Error too large: " << error);
  }
  if (nb_iter == 100) return true;
  return false;
  
}



int main(void) {

  srand(8826);

  gmm::set_warning_level(1);

  for (int iter = 0; iter < 100000; ++iter) {

    try {

      gmm::col_matrix<std::vector<std::complex<double> > >  param0(20, 27);
      gmm::size_type param_u0 [13] = {0 , 1 , 3 , 5 , 6 , 9 , 10 , 11 , 13 , 15 , 20 , 22 , 24};
      gmm::fill_random(param0, 0.2);
      gmm::row_matrix<gmm::slvector<std::complex<double> > >  param1(13, 28);
      gmm::fill_random(param1, 0.2);
      gmm::row_matrix<std::vector<std::complex<double> > >  param2(16, 17);
      gmm::fill_random(param2);
    

      bool ret = test_procedure(gmm::sub_matrix(param0, gmm::sub_interval(0, 20), gmm::sub_index(&param_u0 [0], &param_u0 [13])), gmm::sub_matrix(param1, gmm::sub_interval(0, 13), gmm::sub_interval(2, 16)), param2);
      if (ret) return 0;

    }
    GMM_STANDARD_CATCH_ERROR;
  }
  return 0;
}

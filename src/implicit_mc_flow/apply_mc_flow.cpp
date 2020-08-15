// OpenMesh
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/Attributes.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

/** Standard Includes */
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

/** Matrix Library **/
#include <gmm/gmm.h>

using std::abort;
using std::exception;
using std::flush;
using std::ifstream;
using std::ios;
using std::map;
using std::ofstream;
using std::string;
using namespace OpenMesh;


//********************** Command Line Parser
inline char* getCmdOption(char** begin, char** end, const std::string& option)
{
    // https://stackoverflow.com/a/868894/1608232
    char** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}
inline bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    // https://stackoverflow.com/a/868894/1608232
    return std::find(begin, end, option) != end;
}
//******************************************************************************


/** Typedefs **/
struct MyTraits : public OpenMesh::DefaultTraits
{
    VertexAttributes(OpenMesh::Attributes::Normal |
                     OpenMesh::Attributes::Color);

    FaceAttributes(OpenMesh::Attributes::Normal);
};

typedef OpenMesh::TriMesh_ArrayKernelT<MyTraits> cmesh_t;
typedef OpenMesh::VPropHandleT<double>           vfloat_handle_t;
typedef OpenMesh::EPropHandleT<double>           efloat_handle_t;

typedef gmm::row_matrix<gmm::wsvector<double> > compressed_matrix_t;
typedef gmm::col_matrix<std::vector<double> >   dense_matrix_t;


template <typename mesh_t>
double vfarea(mesh_t& mesh, typename mesh_t::VertexHandle vh)
{
    typedef typename mesh_t::Point          p_t;
    typedef typename mesh_t::VertexHandle   vh_t;
    typedef typename mesh_t::HalfedgeHandle hh_t;
    typedef typename mesh_t::Scalar         scalar_t;

    typename mesh_t::VertexFaceIter vf_it;
    typename mesh_t::FaceVertexIter fv_it;

    double area = 0.0;
    for (vf_it = mesh.vf_iter(vh); vf_it; ++vf_it) {
        fv_it = mesh.fv_iter(vf_it);

        const p_t& P = mesh.point(fv_it);
        ++fv_it;
        const p_t& Q = mesh.point(fv_it);
        ++fv_it;
        const p_t& R = mesh.point(fv_it);

        area += ((Q - P) % (R - P)).norm() * 0.5f;
    }
    return fabs(area) > std::numeric_limits<double>::min() ? area : 1e-8;
}

template <typename mesh_t>
double varea(mesh_t& mesh, typename mesh_t::VertexHandle vh)
{
    typedef typename mesh_t::Point          p_t;
    typedef typename mesh_t::VertexHandle   vh_t;
    typedef typename mesh_t::HalfedgeHandle hh_t;
    typedef typename mesh_t::Scalar         scalar_t;

    typename mesh_t::VertexFaceIter vf_it;
    typename mesh_t::FaceVertexIter fv_it;

    double area = 0.0;
    for (vf_it = mesh.vf_iter(vh); vf_it; ++vf_it) {
        fv_it = mesh.fv_iter(vf_it);

        const p_t& P = mesh.point(fv_it);
        ++fv_it;
        const p_t& Q = mesh.point(fv_it);
        ++fv_it;
        const p_t& R = mesh.point(fv_it);

        area += ((Q - P) % (R - P)).norm() * 0.5f * 0.3333f;
    }
    return fabs(area) > std::numeric_limits<double>::min() ? area : 1e-8;
}

/*
template <typename mesh_t>
m4_mesh_t::Point normalized_mean_curvature_flow(m4_mesh_t& mesh,
m4_mesh_t::VertexHandle vh) { typedef m4_mesh_t::Point p_t; typedef
m4_mesh_t::VertexHandle vh_t; typedef m4_mesh_t::HalfedgeHandle hh_t; typedef
m4_mesh_t::Scalar scalar_t;

  p_t p0, p1, p2, d0, d1;
  vh_t v0, v1;
  hh_t h0, h1, h2;
  scalar_t w, area0, area1, b(0.99);

  size_t valence = 0;

  p0 = mesh.point(vh);
  p_t cog(0.0, 0.0, 0.0);
  v0 = vh;
  area0 = varea(mesh, vh);
  double total_weight = 0.0;
  m4_mesh_t::VertexOHalfedgeIter voh_it;
  for (voh_it = mesh.voh_iter(vh); voh_it; ++voh_it) {
    w = 0.0;

    v1 = mesh.to_vertex_handle(voh_it);
    p1 = mesh.point(v1);

    h0 = voh_it.handle();
    h1 = mesh.opposite_halfedge_handle(voh_it.handle());


    if (!mesh.is_boundary(h0)) {
      h2 = mesh.next_halfedge_handle(h0);
      p2 = mesh.point(mesh.to_vertex_handle(h2));
      d0 = (p0 - p2).normalize();
      d1 = (p1 - p2).normalize();
      w += 1.0 / tan(acos(std::max(-b, std::min(b, (d0|d1)))));
    }

    if (!mesh.is_boundary(h1)) {
      h2 = mesh.next_halfedge_handle(h1);
      p2 = mesh.point(mesh.to_vertex_handle(h2));
      d0 = (p0 - p2).normalize();
      d1 = (p1 - p2).normalize();
      w += 1.0 / tan(acos(std::max(-b, std::min(b, (d0|d1)))));
    }
    w = std::max(static_cast<double>(w), 0.0);
    total_weight += w;
    //area1 = varea(mesh, v1);
    cog += (w) * (p0 - p1);
    ++valence;
  }
  std::cerr << "cog = " << cog << "\n";
  //    return (p0 - cog) * (1.0 / 2.0 * area0);
  return cog * (1.0 / total_weight);
}
*/

template <typename mesh_t>
double mesh_volume(mesh_t& mesh)
{
    typedef typename mesh_t::HalfedgeHandle hh_t;
    typedef typename mesh_t::VertexHandle   vh_t;
    typedef typename mesh_t::Point          p_t;

    double                    volume = 0.0;
    double                    three_inv = 1.0 / 3.0;
    typename mesh_t::FaceIter f_it;
    for (f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        hh_t hh = mesh.halfedge_handle(f_it);
        hh_t hhn = mesh.next_halfedge_handle(hh);

        vh_t v0(mesh.from_vertex_handle(hh));
        vh_t v1(mesh.to_vertex_handle(hh));
        vh_t v2(mesh.to_vertex_handle(hhn));

        p_t p0(mesh.point(v0));
        p_t p1(mesh.point(v1));
        p_t p2(mesh.point(v2));

        p_t g = (p0 + p1 + p2) * three_inv;
        p_t n = (p1 - p0) % (p2 - p0);
        volume += (g | n);
    }

    volume *= 1.0 / 6.0;
    return volume;
}

template <typename mesh_t>
void implicit_fairing_curvature_flow(mesh_t&      mesh,
                                     const size_t n,
                                     const float  dt)
{
    typedef typename mesh_t::Point          p_t;
    typedef typename mesh_t::VertexHandle   vh_t;
    typedef typename mesh_t::HalfedgeHandle hh_t;
    typedef typename mesh_t::Scalar         scalar_t;

    p_t      p0, p1, p2, d0, d1;
    vh_t     v0, v1;
    hh_t     h0, h1, h2;
    scalar_t w, area0, area1, b(0.99);

    size_t valence = 0;

    std::vector<double> B0(mesh.n_vertices());
    std::vector<double> B1(mesh.n_vertices());
    std::vector<double> B2(mesh.n_vertices());

    std::vector<double> X0(mesh.n_vertices());
    std::vector<double> X1(mesh.n_vertices());
    std::vector<double> X2(mesh.n_vertices());

    gmm::row_matrix<gmm::wsvector<double> > K(mesh.n_vertices(),
                                              mesh.n_vertices());
    gmm::row_matrix<gmm::wsvector<double> > A(mesh.n_vertices(),
                                              mesh.n_vertices());
    // gmm::row_matrix< gmm::wsvector<double> > Ai(mesh.n_vertices(),
    // mesh.n_vertices());

    //  size_t num_iter = n;
    //  for (size_t i = 0; i < num_iter; ++i) {
    double vol_init = mesh_volume(mesh);
    gmm::clean(K, 1e-12);
    gmm::clean(A, 1e-12);
    // gmm::clean(Ai, 1e-12);
    // p0 = mesh.point(vh);
    // p_t cog(0.0, 0.0, 0.0);
    // double total_weight = 0.0;

    typename mesh_t::VertexIter v_it;
    for (v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
        size_t idx = v_it.handle().idx();
        A(idx, idx) = vfarea(mesh, v_it);
        // Ai(idx, idx) = 1.0 / (vfarea(mesh, v_it));
        K(idx, idx) = A(idx, idx);
        B0[idx] = A(idx, idx) * static_cast<double>(mesh.point(v_it)[0]);
        B1[idx] = A(idx, idx) * static_cast<double>(mesh.point(v_it)[1]);
        B2[idx] = A(idx, idx) * static_cast<double>(mesh.point(v_it)[2]);

        X0[idx] = 0.0;
        X1[idx] = 0.0;
        X2[idx] = 0.0;
    }

    typename mesh_t::EdgeIter e_it;
    for (e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it) {
        w = 0;
        h0 = mesh.halfedge_handle(e_it, 0);
        h1 = mesh.halfedge_handle(e_it, 1);
        v0 = mesh.from_vertex_handle(h0);
        v1 = mesh.to_vertex_handle(h0);
        p0 = mesh.point(v0);
        p1 = mesh.point(v1);

        if (!mesh.is_boundary(h0)) {
            h2 = mesh.next_halfedge_handle(h0);
            p2 = mesh.point(mesh.to_vertex_handle(h2));
            d0 = (p0 - p2).normalize();
            d1 = (p1 - p2).normalize();
            w += 1.0 / tan(acos(std::max(-b, std::min(b, (d0 | d1)))));
        }

        if (!mesh.is_boundary(h1)) {
            h2 = mesh.next_halfedge_handle(h1);
            p2 = mesh.point(mesh.to_vertex_handle(h2));
            d0 = (p0 - p2).normalize();
            d1 = (p1 - p2).normalize();
            w += 1.0 / tan(acos(std::max(-b, std::min(b, (d0 | d1)))));
        }
        w = std::max(static_cast<double>(w), 0.0);

        w *= 0.25 * dt;
        K(v0.idx(), v1.idx()) = -w;
        K(v1.idx(), v0.idx()) = -w;
        K(v0.idx(), v0.idx()) += w;
        K(v1.idx(), v1.idx()) += w;
    }

    gmm::csc_matrix<double> CK;
    gmm::copy(K, CK);

    // computation of a preconditioner (ILDLT)
    gmm::ildlt_precond<gmm::csc_matrix<double> > P(CK);


    // Solve for x
    {
        gmm::iteration iter(
            1e-8);  // defines an iteration object, with a max residu of 1E-8
        iter.set_noisy(1);
        iter.set_maxiter(5000);
        gmm::bicgstab(K, X0, B0, P, iter);
    }

    // Solve for y
    {
        gmm::iteration iter(
            1e-8);  // defines an iteration object, with a max residu of 1E-8
        iter.set_noisy(1);
        iter.set_maxiter(5000);
        gmm::bicgstab(K, X1, B1, P, iter);
    }

    // Solve for z
    {
        gmm::iteration iter(
            1e-8);  // defines an iteration object, with a max residu of 1E-8
        iter.set_noisy(1);
        iter.set_maxiter(5000);
        gmm::bicgstab(K, X2, B2, P, iter);
    }

    // Copy over the new coordinates
    for (v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
        size_t                 idx = v_it.handle().idx();
        typename mesh_t::Point p(X0[idx], X1[idx], X2[idx]);
        mesh.set_point(v_it, p);
    }

    double vol_final = mesh_volume(mesh);

    double beta = pow((vol_init / vol_final), (1.0 / 3.0));
    std::cerr << "init volume = " << vol_init
              << ", final volume = " << vol_final
              << ", scaling factor = " << beta << "\n";
    for (v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
        typename mesh_t::Point p(mesh.point(v_it));
        mesh.set_point(v_it, p * beta);
    }
    //  }
}


int main(int argc, char** argv)
{

    string mesh_input_filename = "";
    string mesh_output_filename = "";
    size_t n = 5;
    float  dt = 0.f;
    if (argc > 1) {
        if (cmdOptionExists(argv, argc + argv, "-h")) {
            std::cout << "\n\nUsage: apply_mc_flow <-option X>\n";
            std::cout << "   -h:      Display this massage and exits\n";
            std::cout << "   -i:      Input mesh file\n";
            std::cout << "   -o:      Input mesh file\n";
            std::cout << "   -n:      Smoothing step. Default is " << n << "\n";
            std::cout << "   -dt:     Time step. Default is " << dt << "\n\n";
            exit(EXIT_SUCCESS);
        }

        if (cmdOptionExists(argv, argc + argv, "-i")) {
            mesh_input_filename =
                std::string(getCmdOption(argv, argv + argc, "-i"));
        }

        if (cmdOptionExists(argv, argc + argv, "-o")) {
            mesh_output_filename =
                std::string(getCmdOption(argv, argv + argc, "-o"));
        }

        if (cmdOptionExists(argv, argc + argv, "-n")) {
            n = atoi(getCmdOption(argv, argv + argc, "-n"));
        }

        if (cmdOptionExists(argv, argc + argv, "-dt")) {
            dt = std::atof(getCmdOption(argv, argv + argc, "-dt"));
        }
    }

    std::cout << "reading model: [" << mesh_input_filename << "] ..\n ";
    cmesh_t               mesh;
    OpenMesh::IO::Options opt;

    opt += OpenMesh::IO::Options::VertexColor;

    if (!OpenMesh::IO::read_mesh(mesh, mesh_input_filename, opt)) {
        std::cout << "Cannot read mesh: " << mesh_input_filename << std::endl;
        exit(EXIT_FAILURE);
    }

    /// Computing Bounding Box
    cmesh_t::ConstVertexIter vIt(mesh.vertices_begin()),
        vEnd(mesh.vertices_end());
    cmesh_t::Point bbMin, bbMax;

    bbMin = bbMax = mesh.point(vIt);

    for (; vIt != vEnd; ++vIt) {
        bbMin.minimize(mesh.point(vIt));
        bbMax.maximize(mesh.point(vIt));
    }

    float ubbox = (bbMax - bbMin).norm() * 0.01f;

    dt *= ubbox;

    std::cout << "Applying " << n << " smoothing steps with dt = " << dt
              << "(1% bbox diagonal = " << ubbox << ")\n";

    for (size_t i = 0; i < n; ++i) {
        implicit_fairing_curvature_flow(mesh, n, dt);
    }

    if (!OpenMesh::IO::write_mesh(mesh, mesh_output_filename, opt)) {
        std::cout << "Cannot write mesh: " << mesh_output_filename << "\n";
        exit(EXIT_FAILURE);
    }

    std::cout << "writing model: [" << mesh_input_filename << "] ... ";


    return EXIT_SUCCESS;
}

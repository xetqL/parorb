
#include <iostream>
#include "median.hpp"
#include <vector>
#include <mpi.h>
#include <array>
#include <algorithm>

#include <boost/geometry.hpp>
#include <yalbb/parallel_utils.hpp>

using namespace std;
using Real = float;

template<int N>
struct Particle {
    static constexpr int ndim = N;
    std::array<Real, N> position;

    friend ostream &operator<<(ostream &os, const Particle &particle) {
        os << std::fixed << std::setprecision(6);
        copy(particle.position.begin(), particle.position.end(), ostream_iterator<Real>(os, " "));
        return os;
    }

};
#include <random>
#include <cstdlib>
template<int N>
struct RandomParticleGenerator {
    Particle<N> operator()(){
        if constexpr(N==3)
            return Particle<N>{{(Real) std::rand() / RAND_MAX,
                                (Real) std::rand() / RAND_MAX,
                                (Real) std::rand() / RAND_MAX}};
        else
            return Particle<N>{{(Real) std::rand() / RAND_MAX,
                                (Real) std::rand() / RAND_MAX}};
    }
};
template<int N>
struct ParticlePositionGetter {
    std::array<Real, N>* operator()(Particle<N>* p){
        return &p->position;
    }
};
#include "orb.hpp"
using namespace orb;
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldsize,r;

    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    using Particle = Particle<2>;

    std::vector<Particle> particles( r == 1? 0 : 20000 );

    MPI_Datatype particle_datatype;
    MPI_Type_contiguous(Particle::ndim, par::get_mpi_type<Real>(), &particle_datatype);
    MPI_Type_commit(&particle_datatype);

    ORBBalancer<Particle::ndim> lb(particle_datatype, MPI_COMM_WORLD);
    generate(begin(particles), end(particles), RandomParticleGenerator<Particle::ndim>());

    parallel_orb<Particle::ndim>(lb, particles,  [](Particle* p){ return &p->position;}, do_migration<Particle>);

    std::ofstream f;
    f.open(std::to_string(r) + ".particles");
    copy(particles.begin(), particles.end(), ostream_iterator< Particle >(f, "\n") );
    f.close();

    /* get neighbors closer than 0.1 */
    auto neigh = lb.get_neighbors(r, 0.1);

    /* lookup the domain belonging to a particle */
    int p;

    lb.lookup_domain( ORBBalancer<Particle::ndim>::point_t(0.00, 0.00), &p);
    lb.lookup_domain( &particles[0], &p, [](Particle* p){ return &p->position;});

    MPI_Finalize();
    return 0;
}


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

template<int N, class T, class GetPosF>
std::pair<array<Real, N>, array<Real, N>> get_dimension_size(vector<T>& p, GetPosF getPosition){
    const size_t size   = p.size();
    const size_t nparts = size / N;
    array<Real, N> max_by_dim;
    array<Real, N> min_by_dim;
    for(auto i =0;i < N; ++i) {
        max_by_dim.at(i) = std::numeric_limits<Real>::lowest();
        min_by_dim.at(i) = std::numeric_limits<Real>::max();
    }
    for(size_t i = 0; i < nparts; ++i) {
        for(auto dim=0; dim < N; ++dim) {
            max_by_dim.at(dim) = std::max(max_by_dim.at(dim), getPosition(&p.at(i*N + dim))->at(dim));
            min_by_dim.at(dim) = std::min(min_by_dim.at(dim), getPosition(&p.at(i*N + dim))->at(dim));
        }
    }
    return { max_by_dim, min_by_dim };
}
template<int N> using Subdomain = std::array<Real, 2*N>;

template<int N>
struct ORBBalancer {

    const MPI_Comm comm;	
    
    ORBBalancer(MPI_Comm comm = MPI_COMM_WORLD): comm(comm){}

    typedef boost::geometry::model::point<Real, N, boost::geometry::cs::cartesian> point_t;
    typedef boost::geometry::model::box<point_t> box_t;
    inline point_t array_to_point(const std::array<Real, N>& pos) {
        if constexpr(N==3)
            return point_t(pos.at(0), pos.at(1), pos.at(2));
        else
            return point_t(pos.at(0), pos.at(1));
    }
    inline box_t domain_to_box(const Subdomain<N>& domain){
        if constexpr(N==3)
            return box_t(point_t(domain.at(0), domain.at(2), domain.at(3)), point_t(domain.at(1), domain.at(3), domain.at(5)));
        else
            return box_t(point_t(domain.at(0), domain.at(2)), point_t(domain.at(1), domain.at(3)));
    }

    std::vector<Subdomain<N>> partitions;
    void lookup_domain(const point_t& p, int* PE) {
        for((*PE) = partitions.size()-1; (*PE) > -1; (*PE)--) {
            if (boost::geometry::covered_by(p, domain_to_box(partitions.at((*PE))))) return;
        }
    }
    void lookup_domain(std::array<Real, N>* pos, int* PE) {
        auto point = array_to_point(*pos);
        lookup_domain(point, PE);
    }
    void lookup_domain(const std::array<Real, N>& pos, int* PE) {
        auto point = array_to_point(pos);
        lookup_domain(point, PE);
    }
    void lookup_domain(const std::array<Real, N>&& pos, int* PE) {
        auto point = array_to_point(pos);
        lookup_domain(point, PE);
    }
    template<class T, class GetPosF>
    void lookup_domain(T* p, int* PE, GetPosF getPosition) {
        lookup_domain(getPosition(p), PE);
    }
    std::vector<int> get_neighbors(int PE, double min_distance) {
        int worldsize;
        std::vector<int> neighbors;
        MPI_Comm_size(comm, &worldsize);
        box_t targeted_box = domain_to_box(partitions.at(PE));
        for(int i = 0; i < worldsize; ++i){
            box_t current_box = domain_to_box(partitions.at(i));
            if(PE != i && std::abs(boost::geometry::distance(targeted_box, current_box)) < min_distance){
                neighbors.push_back(i);
            }
        }
        return neighbors;
    }
};

template<int N, class T, class GetPosF>
void orb(ORBBalancer<N>& lb, vector<T>& elements, Subdomain<N> box, unsigned int P, GetPosF getPosition, MPI_Datatype datatype, MPI_Comm comm) {
    const size_t size   = elements.size();
    const size_t nparts = size / N;

    auto ncuts = std::log(P) / std::log(2);
    std::vector< Subdomain<N> > domains;
    std::vector< std::vector<T>  > domains_data { std::move(elements) };
    domains.emplace_back( box );
    while(ncuts--) {
        std::vector< Subdomain<N> > subdomains; subdomains.reserve(2*domains.size());
        std::vector< std::vector<T>  > subdomains_data; subdomains_data.reserve(2*domains_data.size());
        //for(const auto& d : domains) {
        const size_t nb_domains = domains.size();
        for(auto id = 0; id < nb_domains; ++id) {
            auto& d = domains.at(id);
            const auto& parent_boundaries = domains.at(id);
            std::vector<T>& elements = domains_data.at(id);
            // Get the largest dimension for current domain
            auto [maxdim, mindim] = get_dimension_size<N, T, GetPosF>(elements, getPosition);
            MPI_Allreduce(MPI_IN_PLACE, maxdim.data(), N, par::get_mpi_type<Real>(), MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, mindim.data(), N, par::get_mpi_type<Real>(), MPI_MIN, MPI_COMM_WORLD);
            for(auto i = 0; i < N; ++i) maxdim.at(i) -= mindim.at(i);
            auto largest_dim = std::distance(begin(maxdim), std::max_element(begin(maxdim), end(maxdim)));
            std::vector<Real> dim_position(elements.size());
            std::transform(elements.begin(), elements.end(), dim_position.begin(),
                                [getPosition, largest_dim](auto& e){ return getPosition(&e)->at(largest_dim); });
            // get the median
            auto median = par::median(dim_position);
            // divide the dataset in two parts: above and below median
            std::array<Real, 2*N> above_boundaries = parent_boundaries, below_boundaries = parent_boundaries;
            above_boundaries.at(2*largest_dim)     = median;
            below_boundaries.at(2*largest_dim + 1) = median;
            subdomains.emplace_back(below_boundaries);
            subdomains.emplace_back(above_boundaries);
            auto median_it = std::partition(elements.begin(), elements.end(),
                                [getPosition, largest_dim, median] (auto& e) { return getPosition(&e)->at(largest_dim) <= median; });
            subdomains_data.emplace_back(std::begin(elements), median_it);
            subdomains_data.emplace_back(median_it, std::end(elements));
        }
        domains = std::move(subdomains);
        domains_data = std::move(subdomains_data);
    }
    do_migration<T>(0, elements, domains_data, datatype, comm);

    lb.partitions = domains;
}

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
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldsize,r;

    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &r);

    using Particle = Particle<2>;

    ORBBalancer<Particle::ndim> lb(MPI_COMM_WORLD);

    std::vector<Particle> particles( r == 1? 0 : 20000 );

    MPI_Datatype particle_datatype;
    MPI_Type_contiguous(Particle::ndim, par::get_mpi_type<Real>(), &particle_datatype);
    MPI_Type_commit(&particle_datatype);

    generate(begin(particles), end(particles), RandomParticleGenerator<Particle::ndim>());

    orb<Particle::ndim>(lb, particles, {0.0, 1.0, 0.0,1.0}, worldsize, [](Particle* p){ return &p->position;}, particle_datatype, MPI_COMM_WORLD);

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

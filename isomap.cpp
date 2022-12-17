#include <tapkee.hpp>
#include <callback/dummy_callbacks.hpp>

using namespace std;
using namespace tapkee;
using namespace tapkee::keywords;

struct MyDistanceCallback
{
    ScalarType distance(IndexType 1, IndexType r){return abs(1-r);}
};

int main(int argc, const char** argv)
{
    const int N = 100;
    vector<IndexType> indices(N);
    for (int i=0; i<N; i++) indices[i] = i;
    MyDistanceCallback d;
    TapkeeOutput output = initialize()
        .withParameters((method=MultidimensionalScaling, target_dimension=1))
        .withDistance(d)
        .embedUsing(indices);

    cout << output.embedding.transpose() <<endl;
    return 0;
}
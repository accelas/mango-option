#include <experimental/mdspan>
#include <vector>
#include <iostream>

int main() {
    std::vector<double> data(12);

    // Create a 3D view: 3x2x2
    namespace stdex = std::experimental;
    stdex::mdspan<double, stdex::extents<size_t, 3, 2, 2>> view(data.data());

    // Fill with test data
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                view[i,j,k] = i*100 + j*10 + k;
            }
        }
    }

    std::cout << "mdspan working! view[1,1,0] = " << view[1,1,0] << "\n";
    return 0;
}

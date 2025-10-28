#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

// ======================================================
// Convolution gaussienne CPU simplifiée
// f_i = sum_j exp(-||x_i - y_j||^2 / (2σ²))
// -----------------------------------------------
// Entrées :
//  - x : tableau N x D (float32)
//  - y : tableau M x D (float32)
//  - sigma : écart-type de la gaussienne (float)
// Sortie :
//  - tableau f de taille N contenant les sommes pour chaque x_i
// ======================================================


py::array_t<float> gaussian_conv(
    py::array_t<float> x,
    py::array_t<float> y,
    float sigma
) {
    // Obtenir les buffers numpy (vérification minimale)
    auto buf_x = x.request(), buf_y = y.request();

    // Dimensions attendues
    const int N = buf_x.shape[0];   
    const int M = buf_y.shape[0];   
    const int D = buf_x.shape[1];   

    // Vérification simple : mêmes dimensions D pour x et y
    if (buf_y.shape[1] != D)
        throw std::runtime_error("x and y must have same dimension");

    // Préparer le tableau résultat (taille N)
    auto result = py::array_t<float>(N);
    auto buf_f = result.request();

    // Pointeurs vers les données brutes (float*)
    const float *px = (float*) buf_x.ptr;
    const float *py = (float*) buf_y.ptr;
    float *pf = (float*) buf_f.ptr;

    // Pré-calcul du coefficient dans l'exponentielle
    const float coef = -1.0f / (2.0f * sigma * sigma);

    // Boucle principale : pour chaque point x_i, calculer la somme sur j
    // Complexité O(N * M * D)
    for (int i = 0; i < N; ++i) {
        float acc = 0.0f;               // accumulateur pour f_i
        for (int j = 0; j < M; ++j) {
            float d2 = 0.0f;            // distance au carré ||x_i - y_j||^2
            // Somme des différences au carré sur les D composantes
            for (int k = 0; k < D; ++k) {
                float diff = px[i*D + k] - py[j*D + k];
                d2 += diff * diff;
            }
            // Ajouter la contribution gaussienne
            acc += std::exp(coef * d2);
        }
        pf[i] = acc; // stocker le résultat pour x_i
    }
    return result;
}

// Liaison Pybind11 : module et exposition de la fonction
PYBIND11_MODULE(keops_bridge, m) {
    m.doc() = "Simplified CPU Gaussian convolution (KeOps-like)";
    m.def("gaussian_conv", &gaussian_conv, "Compute Gaussian convolution");
}

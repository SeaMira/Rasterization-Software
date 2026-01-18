#pragma once

// Este archivo solo debe incluirse en compilaciones CUDA
#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <glm/glm.hpp>

__device__ __constant__ float3 backgroundColor = {0.5f, 0.5f, 0.5f};
__device__ __constant__ float3 atomsColor = {0.8f, 0.1f, 0.1f};
__device__ __constant__ float3 bondsColor = {0.1f, 0.8f, 0.1f};
__device__ __constant__ float diffuse = 0.9f;

// Colores CPK estándar en formato float3 (RGB normalizado)
// Índice = número atómico - 1
// __constant__ float3 CPK_COLORS[40] = {
//     /* 1: H  */ make_float3(1.0f, 1.0f, 1.0f),   // Blanco
//     /* 2: He */ make_float3(0.85f, 1.0f, 1.0f),  // Azul claro
//     /* 3: Li */ make_float3(0.8f, 0.5f, 1.0f),   // Violeta
//     /* 4: Be */ make_float3(0.0f, 1.0f, 0.0f),   // Verde
//     /* 5: B  */ make_float3(1.0f, 0.7f, 0.7f),   // Rosa claro
//     /* 6: C  */ make_float3(0.2f, 0.2f, 0.2f),   // Negro/Gris
//     /* 7: N  */ make_float3(0.0f, 0.0f, 1.0f),   // Azul
//     /* 8: O  */ make_float3(1.0f, 0.0f, 0.0f),   // Rojo
//     /* 9: F  */ make_float3(0.0f, 1.0f, 0.0f),   // Verde
//     /* 10: Ne*/ make_float3(0.7f, 0.9f, 0.96f),  // Azul cielo
//     /* 11: Na*/ make_float3(0.0f, 0.0f, 0.5f),   // Azul oscuro
//     /* 12: Mg*/ make_float3(0.0f, 1.0f, 0.0f),   // Verde
//     /* 13: Al*/ make_float3(0.75f, 0.65f, 0.65f),// Gris rosado
//     /* 14: Si*/ make_float3(0.94f, 0.78f, 0.63f),// Marrón claro
//     /* 15: P */ make_float3(1.0f, 0.5f, 0.0f),   // Naranja
//     /* 16: S */ make_float3(1.0f, 1.0f, 0.0f),   // Amarillo
//     /* 17: Cl*/ make_float3(0.0f, 1.0f, 0.0f),   // Verde
//     /* 18: Ar*/ make_float3(0.5f, 0.8f, 0.9f),   // Azul verdoso
//     /* 19: K */ make_float3(0.5f, 0.0f, 0.5f),   // Violeta
//     /* 20: Ca*/ make_float3(0.6f, 0.6f, 0.6f),   // Gris claro
//     /* 21: Sc*/ make_float3(0.9f, 0.9f, 0.9f),   // Gris plateado
//     /* 22: Ti*/ make_float3(0.75f, 0.76f, 0.78f),// Gris metálico
//     /* 23: V */ make_float3(0.65f, 0.65f, 0.65f),// Gris
//     /* 24: Cr*/ make_float3(0.54f, 0.6f, 0.78f), // Azul grisáceo
//     /* 25: Mn*/ make_float3(0.61f, 0.48f, 0.78f),// Violeta grisáceo
//     /* 26: Fe*/ make_float3(0.8f, 0.4f, 0.0f),   // Marrón
//     /* 27: Co*/ make_float3(0.94f, 0.56f, 0.62f),// Rosa fuerte
//     /* 28: Ni*/ make_float3(0.31f, 0.82f, 0.31f),// Verde esmeralda
//     /* 29: Cu*/ make_float3(0.78f, 0.5f, 0.2f),  // Naranja cobre
//     /* 30: Zn*/ make_float3(0.5f, 0.5f, 0.5f),   // Gris
//     /* 31: Ga*/ make_float3(0.76f, 0.56f, 0.56f),// Rosa grisáceo
//     /* 32: Ge*/ make_float3(0.4f, 0.56f, 0.56f), // Azul grisáceo
//     /* 33: As*/ make_float3(0.74f, 0.5f, 0.89f), // Violeta claro
//     /* 34: Se*/ make_float3(1.0f, 0.63f, 0.0f),  // Naranja fuerte
//     /* 35: Br*/ make_float3(0.6f, 0.2f, 0.2f),   // Marrón rojizo
//     /* 36: Kr*/ make_float3(0.36f, 0.72f, 0.82f),// Azul verdoso
//     /* 37: Rb*/ make_float3(0.44f, 0.18f, 0.69f),// Violeta oscuro
//     /* 38: Sr*/ make_float3(0.0f, 1.0f, 0.0f),   // Verde
//     /* 39: Y */ make_float3(0.58f, 1.0f, 1.0f),  // Azul claro
//     /* 40: Zr*/ make_float3(0.58f, 0.88f, 0.88f)// Azul grisáceo
// };
#endif // __CUDACC__

//
// Created by wpj on 2018/12/18.
//
#ifndef COMPLEX_MODEL
#define COMPLEX_MODEL
#include"model.h"
class ComplEx : public Model {
    int nh;

public:
    ComplEx(int ne, int nr, int nh, double eta, double gamma) : Model(eta, gamma) {
        assert( nh % 2 == 0 );
        this->nh = nh;

        E = uniform_matrix(ne, nh, -init_b, init_b);
        R = uniform_matrix(nr, nh, -init_b, init_b);
        E_g = const_matrix(ne, nh, init_e);
        R_g = const_matrix(nr, nh, init_e);
    }

    double score(int s, int r, int o) const {
        double dot = 0;

        int nh_2 = nh/2;
        for (int i = 0; i < nh_2; i++) {
            dot += R[r][i]      * E[s][i]      * E[o][i];
            dot += R[r][i]      * E[s][nh_2+i] * E[o][nh_2+i];
            dot += R[r][nh_2+i] * E[s][i]      * E[o][nh_2+i];
            dot -= R[r][nh_2+i] * E[s][nh_2+i] * E[o][i];
        }
        return dot;
    }

    void score_grad(
            int s,
            int r,
            int o,
            vector<double>& d_s,
            vector<double>& d_r,
            vector<double>& d_o) {

        int nh_2 = nh/2;
        for (int i = 0; i < nh_2; i++) {
            // re
            d_s[i] = R[r][i] * E[o][i] + R[r][nh_2+i] * E[o][nh_2+i];
            d_r[i] = E[s][i] * E[o][i] + E[s][nh_2+i] * E[o][nh_2+i];
            d_o[i] = R[r][i] * E[s][i] - R[r][nh_2+i] * E[s][nh_2+i];
            // im
            d_s[nh_2+i] = R[r][i] * E[o][nh_2+i] - R[r][nh_2+i] * E[o][i];
            d_r[nh_2+i] = E[s][i] * E[o][nh_2+i] - E[s][nh_2+i] * E[o][i];
            d_o[nh_2+i] = R[r][i] * E[s][nh_2+i] + R[r][nh_2+i] * E[s][i];
        }
    }
};

#endif
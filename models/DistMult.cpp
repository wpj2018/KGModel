//
// Created by wpj on 2018/12/18.
//

#include"model.h"

class DistMult : public Model {
public:
    DistMult(int ne, int nr, int nh, double eta, double gamma) : Model(eta, gamma) {
        this->nh = nh;

        E = uniform_matrix(ne, nh, -init_b, init_b);
        R = uniform_matrix(nr, nh, -init_b, init_b);
        E_g = const_matrix(ne, nh, init_e);
        R_g = const_matrix(nr, nh, init_e);
    }

    double score(int s, int r, int o) const {
        double dot = 0;
        for (int i = 0; i < nh; i++)
            dot += E[s][i] * R[r][i] * E[o][i];
        return dot;
    }

    void score_grad(
            int s,
            int r,
            int o,
            vector<double>& d_s,
            vector<double>& d_r,
            vector<double>& d_o) {

        for (int i = 0; i < nh; i++) {
            d_s[i] = R[r][i] * E[o][i];
            d_r[i] = E[s][i] * E[o][i];
            d_o[i] = E[s][i] * R[r][i];
        }
    }
};



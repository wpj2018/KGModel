//
// Created by wpj on 2018/12/19.
//
#ifndef TRANSH_MODEL
#define TRANSH_MODEL
#include"model.h"
class TransH: public Model{
public:
    TransH(int ne, int nr, int nh,  double eta, double gamma) : Model(eta, gamma) {
        this->nh = nh;

        E = uniform_matrix(ne, nh, -6.0/sqrt(nh), 6.0/sqrt(nh));
        R = uniform_matrix(nr, nh, -6.0/sqrt(nh), 6.0/sqrt(nh));
        A.resize(1);
        A[0] = uniform_matrix(nr, nh, -6.0/sqrt(nh), 6.0/sqrt(nh));

        for(unsigned int i=0;i<E.size();i++){
            l2_normalize(E[i]);
        }

        for(unsigned int i=0;i<R.size();i++){
            l2_normalize(R[i]);
        }

        for(unsigned int i=0;i<A[0].size();i++){
            l2_normalize(A[0][i]);
        }

        E_g = const_matrix(ne, nh, init_e);
        R_g = const_matrix(nr, nh, init_e);
        A_g.resize(1);
        A_g[0] = const_matrix(nr, nh, init_e);
    }
    double score(int s, int r, int o) const{

        double tmp1=0,tmp2=0;
        for (int i=0; i<nh; i++)
        {
            tmp1+=A[0][r][i]*E[s][i];
            tmp2+=A[0][r][i]*E[o][i];
        }

        bool L1 = true;
        double sum=0;
        for (int i=0; i<nh; i++) {
            double diff = fabs(E[s][i] - tmp1 * A[0][r][i] + R[r][i] - (E[o][i] - tmp2 * A[0][r][i]));
            if (L1) {
                sum += diff;
            } else {
                sum += diff * diff;
            }
        }
        return -sum;
    }
    void score_grad(
            int s,
            int r,
            int o,
            vector<double>& d_s,
            vector<double>& d_r,
            vector<double>& d_o,
            vector<double>& d_a) {

        double tmp1=0,tmp2=0;
        for (int i=0; i<nh; i++)
        {
            tmp1+=A[0][r][i]*E[s][i];
            tmp2+=A[0][r][i]*E[o][i];
        }

        bool L1 = true;

        for (int i = 0; i < nh; i++) {
            double x = 2*(E[s][i] - tmp1 * A[0][r][i] + R[r][i] - (E[o][i] - tmp2 * A[0][r][i]));
            if(L1) {
                x = x > 0 ? 1:-1;
            }
            d_s[i] = x * (1-A[0][r][i]*E[s][i]);
            d_r[i] = x;
            d_o[i] = -x * (1-A[0][r][i]*E[o][i]);
            d_a[i] = x * (tmp2-tmp1 + (E[o][i]-E[s][i])*A[0][r][i]);
        }
    }


    void sgd_update(
            int s,
            int r,
            int o,
            const vector<double>& d_s,
            const vector<double>& d_r,
            const vector<double>& d_o,
            const vector<double>& d_a,
            int flag) {

        Model::sgd_update(s, r, o, d_s, d_r, d_o, flag);

        for (int i = 0; i < nh; i++) A[0][r][i] -= flag * eta * d_a[i];

    }

    void adagrad_update(
            int s,
            int r,
            int o,
            const vector<double>& d_s,
            const vector<double>& d_r,
            const vector<double>& d_o,
            const vector<double>& d_a,
            int flag) {

        Model::adagrad_update(s, r, o, d_s, d_r, d_o, flag);

        for (int i = 0; i < nh; i++){
            A_g[0][r][i] += d_a[i] * d_a[i];
            A[0][r][i] -= flag * eta * d_a[i] / sqrt(A_g[0][r][i]);
        }
    }

    void othrogonal_update(int r){

        while (true)
        {
            l2_normalize(A[0][r]);
            double sum=0;
            for (int i=0; i<nh; i++)
            {
                sum+=A[0][r][i]*R[r][i];
            }
            if (sum>0.1)
            {
                for (int i=0; i<nh; i++)
                {
                    R[r][i] -= eta*A[0][r][i];
                    A[0][r][i] -= eta*R[r][i];
                }
            }
            else
                break;
        }
        l2_normalize(A[0][r]);
    }

    void train(int s, int r, int o, int ss, int rr, int oo) {
        vector<double> d_s;
        vector<double> d_r;
        vector<double> d_o;
        vector<double> d_a;

        vector<double> d_ss;
        vector<double> d_rr;
        vector<double> d_oo;
        vector<double> d_aa;

        d_s.resize(E[s].size());
        d_r.resize(R[r].size());
        d_o.resize(E[o].size());
        d_a.resize(A[0][r].size());

        d_ss.resize(E[ss].size());
        d_rr.resize(R[rr].size());
        d_oo.resize(E[oo].size());
        d_aa.resize(A[0][rr].size());

        double sum1 = -score(s, r, o);
        double sum2 = -score(ss, rr, oo);
        double margin = 1.0;
        if (sum1+margin > sum2)
        {
            score_grad(s, r, o, d_s, d_r, d_o, d_a);
            score_grad(ss, rr, oo, d_ss, d_rr, d_oo, d_aa);

            adagrad_update(s, r, o, d_s, d_r, d_o, d_a, 1);
            adagrad_update(ss, rr, oo, d_ss, d_rr, d_oo, d_aa, -1);

            l2_normalize(E[s]);
            l2_normalize(E[o]);

            if (ss != s)
                l2_normalize(E[ss]);

            if (oo != o)
                l2_normalize(E[oo]);

            othrogonal_update(r);

            if (rr != r)
                othrogonal_update(rr);
        }
    }

};
#endif
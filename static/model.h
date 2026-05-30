/**
 * model.h — XGBoost optimizado para STM32
 * GridSearchCV params: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200, 'subsample': 0.7}
 * Accuracy: 75.82%
 * 200 árboles, 15 features
 * Clases: {0: 'extension', 1: 'flexion'}
 * Generado por xgboost.py
 */

#ifndef MODEL_H
#define MODEL_H

#define N_FEATURES     15U
#define N_CLASSES      2U
#define N_TREES_TOTAL  200U
#define N_TREES_CLASS  100U

#define CLASS_EXTENSION  0
#define CLASS_FLEXION  1

static const int SEL_IDX[N_FEATURES] = {0, 1, 2, 3, 4, 7, 8, 10, 13, 14, 18, 22, 26, 27, 28};

#define MEAN_MAV                    1.81241786e+00f
#define MEAN_WL                     4.46959114e+01f
#define MEAN_WAMP                   3.83148232e+01f
#define MEAN_VAR                    2.43982471e+03f
#define MEAN_SNR                    3.20577202e+01f
#define MEAN_LOG                    1.66100609e+00f
#define MEAN_SSI                    3.68921219e+05f
#define MEAN_TFR_MeanEnergy         4.30457382e+01f
#define MEAN_E_Total                2.77059400e+07f
#define MEAN_E_Low                  3.38965100e+06f
#define MEAN_R_Mid                  4.38223884e-04f
#define MEAN_TopFreq2               2.19786301e+01f
#define MEAN_SpectralEntropy        2.92923152e-02f
#define MEAN_SpectralKurtosis       3.85726062e+05f
#define MEAN_MeanFreq               5.75398803e-01f

#define SCALE_MAV                   4.04538631e+00f
#define SCALE_WL                    1.18485205e+03f
#define SCALE_WAMP                  3.74860001e+01f
#define SCALE_VAR                   6.73164062e+04f
#define SCALE_SNR                   8.65129757e+00f
#define SCALE_LOG                   1.11445628e-01f
#define SCALE_SSI                   1.01672490e+07f
#define SCALE_TFR_MeanEnergy        1.22279883e+03f
#define SCALE_E_Total               7.62695168e+08f
#define SCALE_E_Low                 9.17906080e+07f
#define SCALE_R_Mid                 8.61702673e-03f
#define SCALE_TopFreq2              3.10571690e+01f
#define SCALE_SpectralEntropy       2.93952078e-01f
#define SCALE_SpectralKurtosis      6.10206688e+05f
#define SCALE_MeanFreq              1.13872166e+01f

static inline void normalize_select(const float *x_all, float *xn) {
    xn[0] = (x_all[SEL_IDX[0]] - MEAN_MAV) / SCALE_MAV;
    xn[1] = (x_all[SEL_IDX[1]] - MEAN_WL) / SCALE_WL;
    xn[2] = (x_all[SEL_IDX[2]] - MEAN_WAMP) / SCALE_WAMP;
    xn[3] = (x_all[SEL_IDX[3]] - MEAN_VAR) / SCALE_VAR;
    xn[4] = (x_all[SEL_IDX[4]] - MEAN_SNR) / SCALE_SNR;
    xn[5] = (x_all[SEL_IDX[5]] - MEAN_LOG) / SCALE_LOG;
    xn[6] = (x_all[SEL_IDX[6]] - MEAN_SSI) / SCALE_SSI;
    xn[7] = (x_all[SEL_IDX[7]] - MEAN_TFR_MeanEnergy) / SCALE_TFR_MeanEnergy;
    xn[8] = (x_all[SEL_IDX[8]] - MEAN_E_Total) / SCALE_E_Total;
    xn[9] = (x_all[SEL_IDX[9]] - MEAN_E_Low) / SCALE_E_Low;
    xn[10] = (x_all[SEL_IDX[10]] - MEAN_R_Mid) / SCALE_R_Mid;
    xn[11] = (x_all[SEL_IDX[11]] - MEAN_TopFreq2) / SCALE_TopFreq2;
    xn[12] = (x_all[SEL_IDX[12]] - MEAN_SpectralEntropy) / SCALE_SpectralEntropy;
    xn[13] = (x_all[SEL_IDX[13]] - MEAN_SpectralKurtosis) / SCALE_SpectralKurtosis;
    xn[14] = (x_all[SEL_IDX[14]] - MEAN_MeanFreq) / SCALE_MeanFreq;
}

static inline float xgb_tree_0(const float *x) {
    if (x[4] < 2.24323034e-01f) {  /* SNR */
        if (x[4] < -8.07947695e-01f) {  /* SNR */
            if (x[2] < 9.78636742e-01f) {  /* WAMP */
                if (x[2] < 2.85044461e-01f) {  /* WAMP */
                    return -3.27198319e-02f;
                } else {
                    return 6.43091127e-02f;
                }
            } else {
                if (x[0] < -1.30397798e-02f) {  /* MAV */
                    return 1.64445847e-01f;
                } else {
                    return 3.69822942e-02f;
                }
            }
        } else {
            if (x[7] < -3.51358950e-02f) {  /* TFR_MeanEnergy */
                if (x[7] < -3.51430550e-02f) {  /* TFR_MeanEnergy */
                    return -2.73341816e-02f;
                } else {
                    return 7.57124349e-02f;
                }
            } else {
                if (x[10] < -4.41158749e-02f) {  /* R_Mid */
                    return -1.14891075e-01f;
                } else {
                    return 6.47130609e-02f;
                }
            }
        }
    } else {
        if (x[9] < -3.62352207e-02f) {  /* E_Low */
            if (x[3] < -3.62441204e-02f) {  /* VAR */
                if (x[2] < -7.82020569e-01f) {  /* WAMP */
                    return -8.17048550e-02f;
                } else {
                    return -1.40969858e-01f;
                }
            } else {
                if (x[7] < -3.51401605e-02f) {  /* TFR_MeanEnergy */
                    return -7.45870173e-02f;
                } else {
                    return 5.02851792e-03f;
                }
            }
        } else {
            if (x[10] < -5.05589508e-02f) {  /* R_Mid */
                if (x[4] < 3.21932882e-01f) {  /* SNR */
                    return -1.05256334e-01f;
                } else {
                    return -1.83508903e-01f;
                }
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return -2.72657908e-02f;
                } else {
                    return -1.32522628e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_1(const float *x) {
    if (x[4] < -6.26511693e-01f) {  /* SNR */
        if (x[0] < -1.05430773e-02f) {  /* MAV */
            if (x[1] < -3.57238576e-02f) {  /* WL */
                if (x[5] < -7.15442121e-01f) {  /* LOG */
                    return 1.20335408e-02f;
                } else {
                    return 1.02725759e-01f;
                }
            } else {
                if (x[4] < -9.05658960e-01f) {  /* SNR */
                    return 1.62586167e-01f;
                } else {
                    return 1.79481413e-02f;
                }
            }
        } else {
            if (x[2] < -4.35224444e-01f) {  /* WAMP */
                return -1.76446691e-01f;
            } else {
                if (x[1] < -3.56959701e-02f) {  /* WL */
                    return -4.96514626e-02f;
                } else {
                    return 6.48525134e-02f;
                }
            }
        }
    } else {
        if (x[12] < -9.18215439e-02f) {  /* SpectralEntropy */
            if (x[5] < 6.72326088e-02f) {  /* LOG */
                if (x[2] < -7.28667319e-01f) {  /* WAMP */
                    return -6.73420280e-02f;
                } else {
                    return -1.28025383e-01f;
                }
            } else {
                if (x[4] < 5.32929420e-01f) {  /* SNR */
                    return -8.48409757e-02f;
                } else {
                    return -1.55376881e-01f;
                }
            }
        } else {
            if (x[8] < -3.62418406e-02f) {  /* E_Total */
                if (x[6] < -3.62466015e-02f) {  /* SSI */
                    return -3.82168405e-02f;
                } else {
                    return 5.34639917e-02f;
                }
            } else {
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return -5.11182360e-02f;
                } else {
                    return -1.40751526e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_2(const float *x) {
    if (x[12] < -8.08255076e-02f) {  /* SpectralEntropy */
        if (x[12] < -9.28981677e-02f) {  /* SpectralEntropy */
            if (x[5] < 5.57241067e-02f) {  /* LOG */
                if (x[2] < -8.08697224e-01f) {  /* WAMP */
                    return -5.94452508e-02f;
                } else {
                    return -1.10532343e-01f;
                }
            } else {
                if (x[6] < -3.62424925e-02f) {  /* SSI */
                    return -1.17032208e-01f;
                } else {
                    return -1.61505014e-01f;
                }
            }
        } else {
            if (x[6] < -3.62433121e-02f) {  /* SSI */
                if (x[6] < -3.62469777e-02f) {  /* SSI */
                    return -8.81010816e-02f;
                } else {
                    return 1.22724352e-02f;
                }
            } else {
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return -9.05808285e-02f;
                } else {
                    return -1.39475107e-01f;
                }
            }
        }
    } else {
        if (x[5] < 2.50376076e-01f) {  /* LOG */
            if (x[1] < -3.65271010e-02f) {  /* WL */
                if (x[5] < -6.27905130e-01f) {  /* LOG */
                    return -3.35703380e-02f;
                } else {
                    return 7.59540126e-02f;
                }
            } else {
                if (x[1] < -3.56551632e-02f) {  /* WL */
                    return 1.05502322e-01f;
                } else {
                    return 1.52439177e-01f;
                }
            }
        } else {
            if (x[14] < -4.68815453e-02f) {  /* MeanFreq */
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return -8.54830351e-03f;
                } else {
                    return -1.12010695e-01f;
                }
            } else {
                if (x[8] < -3.62323113e-02f) {  /* E_Total */
                    return 1.17652051e-01f;
                } else {
                    return -2.05127485e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_3(const float *x) {
    if (x[12] < -8.56481493e-02f) {  /* SpectralEntropy */
        if (x[7] < -3.51381153e-02f) {  /* TFR_MeanEnergy */
            if (x[12] < -9.45062563e-02f) {  /* SpectralEntropy */
                if (x[2] < -7.28667319e-01f) {  /* WAMP */
                    return -6.17088750e-02f;
                } else {
                    return -1.36610493e-01f;
                }
            } else {
                if (x[7] < -3.51402126e-02f) {  /* TFR_MeanEnergy */
                    return -5.69119118e-02f;
                } else {
                    return 9.18938499e-03f;
                }
            }
        } else {
            if (x[12] < -9.57963988e-02f) {  /* SpectralEntropy */
                if (x[8] < -3.62410471e-02f) {  /* E_Total */
                    return -1.24883272e-01f;
                } else {
                    return -1.56069919e-01f;
                }
            } else {
                if (x[8] < -3.62429693e-02f) {  /* E_Total */
                    return -6.31232336e-02f;
                } else {
                    return -1.05049171e-01f;
                }
            }
        }
    } else {
        if (x[5] < 1.88235059e-01f) {  /* LOG */
            if (x[1] < -3.65271010e-02f) {  /* WL */
                if (x[6] < -3.62462178e-02f) {  /* SSI */
                    return -1.20452074e-02f;
                } else {
                    return 7.38430396e-02f;
                }
            } else {
                if (x[1] < -3.55742276e-02f) {  /* WL */
                    return 1.00606613e-01f;
                } else {
                    return 1.45317420e-01f;
                }
            }
        } else {
            if (x[12] < -3.09275351e-02f) {  /* SpectralEntropy */
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return -2.25384124e-02f;
                } else {
                    return -1.05742417e-01f;
                }
            } else {
                if (x[0] < -1.30397798e-02f) {  /* MAV */
                    return 1.06297567e-01f;
                } else {
                    return 1.68955605e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_4(const float *x) {
    if (x[12] < -6.88733160e-02f) {  /* SpectralEntropy */
        if (x[12] < -9.10081789e-02f) {  /* SpectralEntropy */
            if (x[7] < -3.51383947e-02f) {  /* TFR_MeanEnergy */
                if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                    return -3.73811908e-02f;
                } else {
                    return -7.59733990e-02f;
                }
            } else {
                if (x[1] < -3.72385234e-02f) {  /* WL */
                    return -1.21404923e-01f;
                } else {
                    return -6.73711002e-02f;
                }
            }
        } else {
            if (x[7] < -3.51372249e-02f) {  /* TFR_MeanEnergy */
                if (x[0] < -4.60015498e-02f) {  /* MAV */
                    return -3.69248763e-02f;
                } else {
                    return 4.39691357e-02f;
                }
            } else {
                if (x[1] < -3.66114378e-02f) {  /* WL */
                    return -6.47409111e-02f;
                } else {
                    return -1.54663011e-01f;
                }
            }
        }
    } else {
        if (x[1] < -3.59068140e-02f) {  /* WL */
            if (x[7] < -3.51358950e-02f) {  /* TFR_MeanEnergy */
                if (x[5] < -6.81855738e-01f) {  /* LOG */
                    return -1.99580342e-02f;
                } else {
                    return 8.57079253e-02f;
                }
            } else {
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return -3.42358882e-03f;
                } else {
                    return -7.21819699e-02f;
                }
            }
        } else {
            if (x[0] < -8.97467416e-03f) {  /* MAV */
                if (x[3] < -3.62439789e-02f) {  /* VAR */
                    return 1.76322758e-02f;
                } else {
                    return 1.26038730e-01f;
                }
            } else {
                if (x[1] < -3.15750279e-02f) {  /* WL */
                    return 3.04718595e-02f;
                } else {
                    return -7.54514858e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_5(const float *x) {
    if (x[12] < -8.11554939e-02f) {  /* SpectralEntropy */
        if (x[12] < -9.36690420e-02f) {  /* SpectralEntropy */
            if (x[5] < 9.77115929e-02f) {  /* LOG */
                if (x[2] < -7.82020569e-01f) {  /* WAMP */
                    return -4.48918156e-02f;
                } else {
                    return -9.86566991e-02f;
                }
            } else {
                if (x[8] < -3.62413265e-02f) {  /* E_Total */
                    return -1.00142710e-01f;
                } else {
                    return -1.36975199e-01f;
                }
            }
        } else {
            if (x[5] < 1.50882408e-01f) {  /* LOG */
                if (x[9] < -3.62664275e-02f) {  /* E_Low */
                    return -3.55974957e-02f;
                } else {
                    return 2.27747243e-02f;
                }
            } else {
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return -5.52558266e-02f;
                } else {
                    return -1.05515733e-01f;
                }
            }
        }
    } else {
        if (x[5] < 3.21623117e-01f) {  /* LOG */
            if (x[1] < -3.62047143e-02f) {  /* WL */
                if (x[6] < -3.62478681e-02f) {  /* SSI */
                    return -1.90260019e-02f;
                } else {
                    return 6.20823689e-02f;
                }
            } else {
                if (x[13] < -6.12878978e-01f) {  /* SpectralKurtosis */
                    return 1.22914016e-01f;
                } else {
                    return 8.16002190e-02f;
                }
            }
        } else {
            if (x[12] < -4.86463681e-02f) {  /* SpectralEntropy */
                if (x[2] < 6.58517241e-01f) {  /* WAMP */
                    return -3.85681093e-02f;
                } else {
                    return -1.13868810e-01f;
                }
            } else {
                if (x[8] < -3.62337753e-02f) {  /* E_Total */
                    return 7.74981752e-02f;
                } else {
                    return -1.10280924e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_6(const float *x) {
    if (x[12] < -6.44787103e-02f) {  /* SpectralEntropy */
        if (x[12] < -9.13207531e-02f) {  /* SpectralEntropy */
            if (x[7] < -3.51384319e-02f) {  /* TFR_MeanEnergy */
                if (x[2] < -7.01990664e-01f) {  /* WAMP */
                    return -3.77762280e-02f;
                } else {
                    return -9.95609164e-02f;
                }
            } else {
                if (x[12] < -9.86877456e-02f) {  /* SpectralEntropy */
                    return -1.38971791e-01f;
                } else {
                    return -8.50041732e-02f;
                }
            }
        } else {
            if (x[7] < -3.51350978e-02f) {  /* TFR_MeanEnergy */
                if (x[7] < -3.51418890e-02f) {  /* TFR_MeanEnergy */
                    return -3.47722471e-02f;
                } else {
                    return 2.39827782e-02f;
                }
            } else {
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return -5.63564412e-02f;
                } else {
                    return -1.20196275e-01f;
                }
            }
        }
    } else {
        if (x[1] < -3.58789302e-02f) {  /* WL */
            if (x[8] < -3.62421833e-02f) {  /* E_Total */
                if (x[7] < -3.51435505e-02f) {  /* TFR_MeanEnergy */
                    return -1.28683923e-02f;
                } else {
                    return 8.36409703e-02f;
                }
            } else {
                if (x[2] < 1.32543290e+00f) {  /* WAMP */
                    return 4.10295324e-03f;
                } else {
                    return -8.31623003e-02f;
                }
            }
        } else {
            if (x[7] < -3.51267308e-02f) {  /* TFR_MeanEnergy */
                if (x[12] < -2.36843526e-02f) {  /* SpectralEntropy */
                    return 2.69382354e-03f;
                } else {
                    return 1.12131000e-01f;
                }
            } else {
                if (x[2] < -5.68607569e-01f) {  /* WAMP */
                    return -1.45498157e-01f;
                } else {
                    return 2.61916649e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_7(const float *x) {
    if (x[2] < -4.08547819e-01f) {  /* WAMP */
        if (x[7] < -3.51380371e-02f) {  /* TFR_MeanEnergy */
            if (x[3] < -3.62441167e-02f) {  /* VAR */
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return -1.23594357e-02f;
                } else {
                    return -4.99915443e-02f;
                }
            } else {
                if (x[8] < -3.62502374e-02f) {  /* E_Total */
                    return -5.18139303e-02f;
                } else {
                    return 1.92448404e-02f;
                }
            }
        } else {
            if (x[14] < -5.03076874e-02f) {  /* MeanFreq */
                if (x[0] < -2.70186439e-02f) {  /* MAV */
                    return -9.15537477e-02f;
                } else {
                    return -1.26664698e-01f;
                }
            } else {
                if (x[2] < -5.15254319e-01f) {  /* WAMP */
                    return -6.72120452e-02f;
                } else {
                    return 4.30843281e-03f;
                }
            }
        }
    } else {
        if (x[0] < -3.08925323e-02f) {  /* MAV */
            if (x[6] < -3.62472236e-02f) {  /* SSI */
                if (x[2] < 1.05866659e+00f) {  /* WAMP */
                    return -9.40325484e-03f;
                } else {
                    return 6.91572949e-02f;
                }
            } else {
                if (x[13] < -6.00743711e-01f) {  /* SpectralKurtosis */
                    return 1.06074587e-01f;
                } else {
                    return 5.99900372e-02f;
                }
            }
        } else {
            if (x[14] < -4.70525660e-02f) {  /* MeanFreq */
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return -1.95706896e-02f;
                } else {
                    return -8.62691179e-02f;
                }
            } else {
                if (x[10] < -5.00858799e-02f) {  /* R_Mid */
                    return -1.57889426e-01f;
                } else {
                    return 7.26667270e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_8(const float *x) {
    if (x[4] < -4.94039655e-01f) {  /* SNR */
        if (x[2] < 1.91231871e+00f) {  /* WAMP */
            if (x[5] < 1.88235059e-01f) {  /* LOG */
                if (x[8] < -3.62512134e-02f) {  /* E_Total */
                    return 8.84615723e-03f;
                } else {
                    return 7.73847476e-02f;
                }
            } else {
                if (x[14] < -4.70525660e-02f) {  /* MeanFreq */
                    return -1.87299233e-02f;
                } else {
                    return 3.93686295e-02f;
                }
            }
        } else {
            if (x[13] < -6.08117580e-01f) {  /* SpectralKurtosis */
                if (x[7] < -3.51309516e-02f) {  /* TFR_MeanEnergy */
                    return 1.19695865e-01f;
                } else {
                    return 5.49440794e-02f;
                }
            } else {
                if (x[4] < -9.05658960e-01f) {  /* SNR */
                    return -2.78492924e-04f;
                } else {
                    return -1.42320871e-01f;
                }
            }
        }
    } else {
        if (x[12] < -9.36690420e-02f) {  /* SpectralEntropy */
            if (x[2] < -7.82020569e-01f) {  /* WAMP */
                if (x[5] < -1.25139896e-02f) {  /* LOG */
                    return -2.82495059e-02f;
                } else {
                    return -8.14106688e-02f;
                }
            } else {
                if (x[10] < -5.04248068e-02f) {  /* R_Mid */
                    return -1.03920154e-01f;
                } else {
                    return 3.22845876e-02f;
                }
            }
        } else {
            if (x[7] < -3.51371020e-02f) {  /* TFR_MeanEnergy */
                if (x[6] < -3.62457298e-02f) {  /* SSI */
                    return -1.71418898e-02f;
                } else {
                    return 2.89690830e-02f;
                }
            } else {
                if (x[2] < 6.85193837e-01f) {  /* WAMP */
                    return -4.53557670e-02f;
                } else {
                    return -1.08724616e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_9(const float *x) {
    if (x[14] < -5.00544272e-02f) {  /* MeanFreq */
        if (x[7] < -3.51383947e-02f) {  /* TFR_MeanEnergy */
            if (x[12] < -9.50429589e-02f) {  /* SpectralEntropy */
                if (x[2] < -8.08697224e-01f) {  /* WAMP */
                    return -2.86852811e-02f;
                } else {
                    return -8.95376876e-02f;
                }
            } else {
                if (x[5] < -1.74734831e-01f) {  /* LOG */
                    return -3.77808437e-02f;
                } else {
                    return 9.93269309e-03f;
                }
            }
        } else {
            if (x[1] < -3.72970141e-02f) {  /* WL */
                if (x[8] < -3.62413265e-02f) {  /* E_Total */
                    return -8.41019079e-02f;
                } else {
                    return -1.21163107e-01f;
                }
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 5.66288317e-03f;
                } else {
                    return -6.74588531e-02f;
                }
            }
        }
    } else {
        if (x[1] < -3.58789302e-02f) {  /* WL */
            if (x[7] < -3.51360068e-02f) {  /* TFR_MeanEnergy */
                if (x[8] < -3.62504050e-02f) {  /* E_Total */
                    return -1.62824001e-02f;
                } else {
                    return 4.46026735e-02f;
                }
            } else {
                if (x[12] < -4.76557165e-02f) {  /* SpectralEntropy */
                    return -6.46000430e-02f;
                } else {
                    return 2.60911044e-03f;
                }
            }
        } else {
            if (x[4] < -8.92963409e-01f) {  /* SNR */
                if (x[7] < -3.51267308e-02f) {  /* TFR_MeanEnergy */
                    return 9.76794213e-02f;
                } else {
                    return -9.14663821e-03f;
                }
            } else {
                if (x[2] < 1.59219909e+00f) {  /* WAMP */
                    return 1.38576820e-01f;
                } else {
                    return -9.64073017e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_10(const float *x) {
    if (x[12] < -7.16465488e-02f) {  /* SpectralEntropy */
        if (x[7] < -3.51371579e-02f) {  /* TFR_MeanEnergy */
            if (x[14] < -5.02071157e-02f) {  /* MeanFreq */
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return -5.14602894e-03f;
                } else {
                    return -4.28460054e-02f;
                }
            } else {
                if (x[0] < -4.19019088e-02f) {  /* MAV */
                    return -1.42917754e-02f;
                } else {
                    return 3.50425653e-02f;
                }
            }
        } else {
            if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                if (x[10] < -5.05589508e-02f) {  /* R_Mid */
                    return -9.42036137e-02f;
                } else {
                    return -4.03137319e-02f;
                }
            } else {
                if (x[5] < 5.57770550e-01f) {  /* LOG */
                    return -8.22103471e-02f;
                } else {
                    return -1.15142740e-01f;
                }
            }
        }
    } else {
        if (x[1] < -3.55742276e-02f) {  /* WL */
            if (x[7] < -3.51358950e-02f) {  /* TFR_MeanEnergy */
                if (x[5] < -6.27905130e-01f) {  /* LOG */
                    return -4.41753259e-03f;
                } else {
                    return 6.11904077e-02f;
                }
            } else {
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return 1.17117567e-02f;
                } else {
                    return -4.31116186e-02f;
                }
            }
        } else {
            if (x[7] < -3.51267308e-02f) {  /* TFR_MeanEnergy */
                if (x[10] < -4.93563786e-02f) {  /* R_Mid */
                    return -4.72248578e-03f;
                } else {
                    return 9.94780287e-02f;
                }
            } else {
                if (x[3] < -3.62434611e-02f) {  /* VAR */
                    return 5.70731349e-02f;
                } else {
                    return -5.04252799e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_11(const float *x) {
    if (x[12] < -6.21565394e-02f) {  /* SpectralEntropy */
        if (x[5] < 3.21623117e-01f) {  /* LOG */
            if (x[12] < -8.87201652e-02f) {  /* SpectralEntropy */
                if (x[12] < -9.55390856e-02f) {  /* SpectralEntropy */
                    return -4.24062423e-02f;
                } else {
                    return -1.61323659e-02f;
                }
            } else {
                if (x[0] < -4.65951823e-02f) {  /* MAV */
                    return -2.26364043e-02f;
                } else {
                    return 2.73997840e-02f;
                }
            }
        } else {
            if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                if (x[12] < -9.26299021e-02f) {  /* SpectralEntropy */
                    return -1.01948477e-01f;
                } else {
                    return -2.38965973e-02f;
                }
            } else {
                if (x[10] < -4.75467108e-02f) {  /* R_Mid */
                    return -1.01112537e-01f;
                } else {
                    return -4.24788967e-02f;
                }
            }
        }
    } else {
        if (x[1] < -3.56157161e-02f) {  /* WL */
            if (x[6] < -3.62436511e-02f) {  /* SSI */
                if (x[5] < -6.81855738e-01f) {  /* LOG */
                    return 1.55394350e-03f;
                } else {
                    return 6.29994497e-02f;
                }
            } else {
                if (x[2] < 1.51216924e+00f) {  /* WAMP */
                    return 1.36396540e-02f;
                } else {
                    return -5.18320166e-02f;
                }
            }
        } else {
            if (x[2] < -6.48637414e-01f) {  /* WAMP */
                return -1.34811446e-01f;
            } else {
                if (x[5] < 1.11175334e+00f) {  /* LOG */
                    return 9.43452939e-02f;
                } else {
                    return 1.29657183e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_12(const float *x) {
    if (x[4] < 5.02185803e-03f) {  /* SNR */
        if (x[1] < -3.59068140e-02f) {  /* WL */
            if (x[9] < -3.62332501e-02f) {  /* E_Low */
                if (x[7] < -3.51456925e-02f) {  /* TFR_MeanEnergy */
                    return -2.50163227e-02f;
                } else {
                    return 4.47103083e-02f;
                }
            } else {
                if (x[10] < -4.65358421e-02f) {  /* R_Mid */
                    return -4.08602655e-02f;
                } else {
                    return 2.51207110e-02f;
                }
            }
        } else {
            if (x[9] < -3.61588970e-02f) {  /* E_Low */
                if (x[3] < -3.62439789e-02f) {  /* VAR */
                    return 5.31366048e-03f;
                } else {
                    return 8.60235319e-02f;
                }
            } else {
                if (x[10] < -4.71174084e-02f) {  /* R_Mid */
                    return -9.15091857e-02f;
                } else {
                    return 1.91104375e-02f;
                }
            }
        }
    } else {
        if (x[7] < -3.51381153e-02f) {  /* TFR_MeanEnergy */
            if (x[10] < -5.04404493e-02f) {  /* R_Mid */
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 3.06152902e-03f;
                } else {
                    return -3.84277515e-02f;
                }
            } else {
                if (x[6] < -3.62455100e-02f) {  /* SSI */
                    return -2.64317375e-02f;
                } else {
                    return 2.68086884e-02f;
                }
            }
        } else {
            if (x[10] < -5.05589508e-02f) {  /* R_Mid */
                if (x[4] < 1.74811363e+00f) {  /* SNR */
                    return -7.28158355e-02f;
                } else {
                    return -1.18911207e-01f;
                }
            } else {
                if (x[1] < -3.69236171e-02f) {  /* WL */
                    return -2.54176743e-02f;
                } else {
                    return -1.07177436e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_13(const float *x) {
    if (x[4] < 1.17012508e-01f) {  /* SNR */
        if (x[1] < -3.58789302e-02f) {  /* WL */
            if (x[5] < 8.70909095e-02f) {  /* LOG */
                if (x[6] < -3.62466015e-02f) {  /* SSI */
                    return -5.67521947e-03f;
                } else {
                    return 4.40110303e-02f;
                }
            } else {
                if (x[2] < 1.48549259e+00f) {  /* WAMP */
                    return -1.67184230e-02f;
                } else {
                    return -9.88876894e-02f;
                }
            }
        } else {
            if (x[8] < -3.62343453e-02f) {  /* E_Total */
                if (x[1] < -3.50219570e-02f) {  /* WL */
                    return 6.13522343e-02f;
                } else {
                    return 1.04734384e-01f;
                }
            } else {
                if (x[2] < -5.68607569e-01f) {  /* WAMP */
                    return -1.27041265e-01f;
                } else {
                    return 7.94609543e-03f;
                }
            }
        }
    } else {
        if (x[5] < 1.29493430e-01f) {  /* LOG */
            if (x[4] < 7.96598792e-01f) {  /* SNR */
                if (x[0] < -4.22697887e-02f) {  /* MAV */
                    return -3.19124199e-02f;
                } else {
                    return 5.17684175e-03f;
                }
            } else {
                if (x[2] < -7.82020569e-01f) {  /* WAMP */
                    return -2.65984125e-02f;
                } else {
                    return -8.80028531e-02f;
                }
            }
        } else {
            if (x[1] < -3.72174382e-02f) {  /* WL */
                if (x[10] < -5.06074429e-02f) {  /* R_Mid */
                    return -9.95919332e-02f;
                } else {
                    return -5.85509725e-02f;
                }
            } else {
                if (x[5] < 1.58213675e+00f) {  /* LOG */
                    return -4.21332270e-02f;
                } else {
                    return 6.62572011e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_14(const float *x) {
    if (x[12] < -6.50291964e-02f) {  /* SpectralEntropy */
        if (x[5] < 6.72326088e-02f) {  /* LOG */
            if (x[13] < -3.16034436e-01f) {  /* SpectralKurtosis */
                if (x[5] < -3.25690001e-01f) {  /* LOG */
                    return -1.19989486e-02f;
                } else {
                    return 3.74483429e-02f;
                }
            } else {
                if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                    return -2.54326710e-03f;
                } else {
                    return -3.38814631e-02f;
                }
            }
        } else {
            if (x[2] < 6.05163991e-01f) {  /* WAMP */
                if (x[12] < -7.56563991e-02f) {  /* SpectralEntropy */
                    return -5.08646555e-02f;
                } else {
                    return 1.49816349e-02f;
                }
            } else {
                if (x[12] < -6.67910427e-02f) {  /* SpectralEntropy */
                    return -1.26566708e-01f;
                } else {
                    return -3.80214117e-02f;
                }
            }
        }
    } else {
        if (x[1] < -3.58238406e-02f) {  /* WL */
            if (x[7] < -3.51358950e-02f) {  /* TFR_MeanEnergy */
                if (x[0] < -5.61225004e-02f) {  /* MAV */
                    return -1.98253933e-02f;
                } else {
                    return 4.26651947e-02f;
                }
            } else {
                if (x[2] < 1.21872640e+00f) {  /* WAMP */
                    return 1.55904042e-02f;
                } else {
                    return -6.03241883e-02f;
                }
            }
        } else {
            if (x[3] < -3.62439752e-02f) {  /* VAR */
                if (x[2] < 1.67222905e+00f) {  /* WAMP */
                    return 1.06631614e-01f;
                } else {
                    return -1.20566428e-01f;
                }
            } else {
                if (x[7] < -3.51297967e-02f) {  /* TFR_MeanEnergy */
                    return 8.27398002e-02f;
                } else {
                    return -1.54884707e-04f;
                }
            }
        }
    }
}

static inline float xgb_tree_15(const float *x) {
    if (x[13] < -5.88590384e-01f) {  /* SpectralKurtosis */
        if (x[9] < -3.61676104e-02f) {  /* E_Low */
            if (x[1] < -3.53919528e-02f) {  /* WL */
                if (x[4] < -1.32085180e+00f) {  /* SNR */
                    return -3.49298380e-02f;
                } else {
                    return 4.48436439e-02f;
                }
            } else {
                if (x[2] < 1.69890559e+00f) {  /* WAMP */
                    return 3.36279757e-02f;
                } else {
                    return 9.14873406e-02f;
                }
            }
        } else {
            if (x[2] < -1.41781554e-01f) {  /* WAMP */
                return -1.42509609e-01f;
            } else {
                if (x[4] < -9.65552390e-01f) {  /* SNR */
                    return 2.58509368e-02f;
                } else {
                    return -7.76443034e-02f;
                }
            }
        }
    } else {
        if (x[12] < -9.53515694e-02f) {  /* SpectralEntropy */
            if (x[2] < -8.35373819e-01f) {  /* WAMP */
                if (x[0] < -3.75898033e-02f) {  /* MAV */
                    return -9.81272757e-03f;
                } else {
                    return -5.98850846e-02f;
                }
            } else {
                if (x[0] < -4.53202799e-02f) {  /* MAV */
                    return 1.95440575e-02f;
                } else {
                    return -8.10726434e-02f;
                }
            }
        } else {
            if (x[5] < 8.18698853e-02f) {  /* LOG */
                if (x[8] < -3.62484790e-02f) {  /* E_Total */
                    return -1.95215475e-02f;
                } else {
                    return 1.95048302e-02f;
                }
            } else {
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return -1.29999248e-02f;
                } else {
                    return -5.41047826e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_16(const float *x) {
    if (x[13] < -5.93904257e-01f) {  /* SpectralKurtosis */
        if (x[7] < -3.51297967e-02f) {  /* TFR_MeanEnergy */
            if (x[1] < -3.53484228e-02f) {  /* WL */
                if (x[5] < -5.49159765e-01f) {  /* LOG */
                    return 3.09510762e-03f;
                } else {
                    return 5.10946177e-02f;
                }
            } else {
                if (x[2] < 1.75225902e+00f) {  /* WAMP */
                    return 3.00992187e-02f;
                } else {
                    return 9.51679125e-02f;
                }
            }
        } else {
            if (x[2] < -1.41781554e-01f) {  /* WAMP */
                return -1.33450717e-01f;
            } else {
                if (x[1] < -3.50219570e-02f) {  /* WL */
                    return -2.62267124e-02f;
                } else {
                    return 5.62770031e-02f;
                }
            }
        }
    } else {
        if (x[1] < -3.72249186e-02f) {  /* WL */
            if (x[7] < -3.51383574e-02f) {  /* TFR_MeanEnergy */
                if (x[2] < -7.28667319e-01f) {  /* WAMP */
                    return -1.49168670e-02f;
                } else {
                    return -6.98066726e-02f;
                }
            } else {
                if (x[4] < 1.62340498e+00f) {  /* SNR */
                    return -5.49657829e-02f;
                } else {
                    return -1.15058325e-01f;
                }
            }
        } else {
            if (x[5] < 3.74677360e-01f) {  /* LOG */
                if (x[10] < -5.01533709e-02f) {  /* R_Mid */
                    return -7.10570905e-03f;
                } else {
                    return 2.25521419e-02f;
                }
            } else {
                if (x[13] < -2.69919932e-01f) {  /* SpectralKurtosis */
                    return -4.35633324e-02f;
                } else {
                    return 1.88424177e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_17(const float *x) {
    if (x[10] < -5.01533709e-02f) {  /* R_Mid */
        if (x[7] < -3.51380780e-02f) {  /* TFR_MeanEnergy */
            if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                if (x[4] < 8.48652959e-01f) {  /* SNR */
                    return 1.13241123e-02f;
                } else {
                    return -1.99852996e-02f;
                }
            } else {
                if (x[10] < -5.04404493e-02f) {  /* R_Mid */
                    return -3.40910293e-02f;
                } else {
                    return 3.85526009e-03f;
                }
            }
        } else {
            if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                if (x[10] < -5.05651496e-02f) {  /* R_Mid */
                    return -5.81524856e-02f;
                } else {
                    return -2.67089847e-02f;
                }
            } else {
                return -1.09757774e-01f;
            }
        }
    } else {
        if (x[10] < -4.66369651e-02f) {  /* R_Mid */
            if (x[5] < 8.70909095e-02f) {  /* LOG */
                if (x[6] < -3.62462550e-02f) {  /* SSI */
                    return -8.41782894e-03f;
                } else {
                    return 4.20377366e-02f;
                }
            } else {
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return -1.57280301e-03f;
                } else {
                    return -4.04478572e-02f;
                }
            }
        } else {
            if (x[2] < 3.38397712e-01f) {  /* WAMP */
                if (x[5] < -4.81943178e+00f) {  /* LOG */
                    return 1.47555158e-01f;
                } else {
                    return -2.27919202e-02f;
                }
            } else {
                if (x[2] < 2.12573171e+00f) {  /* WAMP */
                    return 4.87093627e-02f;
                } else {
                    return 9.69242826e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_18(const float *x) {
    if (x[12] < -6.38926774e-02f) {  /* SpectralEntropy */
        if (x[5] < 3.74677360e-01f) {  /* LOG */
            if (x[1] < -3.71847898e-02f) {  /* WL */
                if (x[5] < -4.81943178e+00f) {  /* LOG */
                    return 1.53490603e-01f;
                } else {
                    return -2.08108593e-02f;
                }
            } else {
                if (x[10] < -4.97110672e-02f) {  /* R_Mid */
                    return -2.78758141e-03f;
                } else {
                    return 2.66331937e-02f;
                }
            }
        } else {
            if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                if (x[1] < -3.72854508e-02f) {  /* WL */
                    return -9.02451128e-02f;
                } else {
                    return -1.98293477e-02f;
                }
            } else {
                if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                    return -6.67668507e-02f;
                } else {
                    return -1.00011982e-01f;
                }
            }
        }
    } else {
        if (x[1] < -3.54348011e-02f) {  /* WL */
            if (x[5] < 1.88235059e-01f) {  /* LOG */
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return -2.30798498e-02f;
                } else {
                    return 3.87409404e-02f;
                }
            } else {
                if (x[14] < -4.86704260e-02f) {  /* MeanFreq */
                    return 5.59215434e-02f;
                } else {
                    return -9.61613469e-03f;
                }
            }
        } else {
            if (x[6] < -3.62357050e-02f) {  /* SSI */
                if (x[10] < -4.93563786e-02f) {  /* R_Mid */
                    return -6.72962889e-02f;
                } else {
                    return 7.62794539e-02f;
                }
            } else {
                if (x[5] < 2.17295122e+00f) {  /* LOG */
                    return -4.04131003e-02f;
                } else {
                    return 8.04447383e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_19(const float *x) {
    if (x[1] < -3.59360613e-02f) {  /* WL */
        if (x[10] < -5.05131334e-02f) {  /* R_Mid */
            if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                if (x[0] < -3.79430652e-02f) {  /* MAV */
                    return 6.01588702e-03f;
                } else {
                    return -3.84292267e-02f;
                }
            } else {
                if (x[2] < -9.95433569e-01f) {  /* WAMP */
                    return 9.94205941e-03f;
                } else {
                    return -4.11415137e-02f;
                }
            }
        } else {
            if (x[8] < -3.62394936e-02f) {  /* E_Total */
                if (x[8] < -3.62528414e-02f) {  /* E_Total */
                    return -2.67370399e-02f;
                } else {
                    return 1.32527892e-02f;
                }
            } else {
                if (x[12] < -6.67910427e-02f) {  /* SpectralEntropy */
                    return -6.46835342e-02f;
                } else {
                    return -1.34939998e-02f;
                }
            }
        }
    } else {
        if (x[4] < -8.92963409e-01f) {  /* SNR */
            if (x[2] < -5.68607569e-01f) {  /* WAMP */
                return -1.08347498e-01f;
            } else {
                if (x[10] < -4.93235514e-02f) {  /* R_Mid */
                    return -2.57656910e-02f;
                } else {
                    return 6.27681911e-02f;
                }
            }
        } else {
            if (x[2] < 1.59219909e+00f) {  /* WAMP */
                if (x[0] < -2.04130169e-02f) {  /* MAV */
                    return 1.03640571e-01f;
                } else {
                    return 5.88735752e-03f;
                }
            } else {
                if (x[14] < -4.59191911e-02f) {  /* MeanFreq */
                    return -5.27196340e-02f;
                } else {
                    return -1.48646951e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_20(const float *x) {
    if (x[12] < -6.38926774e-02f) {  /* SpectralEntropy */
        if (x[7] < -3.51355672e-02f) {  /* TFR_MeanEnergy */
            if (x[1] < -3.72249186e-02f) {  /* WL */
                if (x[2] < -7.82020569e-01f) {  /* WAMP */
                    return -1.08339069e-02f;
                } else {
                    return -4.95625995e-02f;
                }
            } else {
                if (x[12] < -9.40425396e-02f) {  /* SpectralEntropy */
                    return -5.06087318e-02f;
                } else {
                    return 1.01604555e-02f;
                }
            }
        } else {
            if (x[14] < -5.03286049e-02f) {  /* MeanFreq */
                if (x[1] < -3.75030935e-02f) {  /* WL */
                    return -2.65791323e-02f;
                } else {
                    return -1.04620099e-01f;
                }
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 3.05106323e-02f;
                } else {
                    return -4.89200950e-02f;
                }
            }
        }
    } else {
        if (x[2] < 1.99234867e+00f) {  /* WAMP */
            if (x[6] < -3.62331904e-02f) {  /* SSI */
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return -6.91387523e-03f;
                } else {
                    return 2.51074471e-02f;
                }
            } else {
                if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                    return -9.25898328e-02f;
                } else {
                    return 2.56071985e-02f;
                }
            }
        } else {
            if (x[14] < -4.50671427e-02f) {  /* MeanFreq */
                return -5.58533669e-02f;
            } else {
                if (x[5] < 1.20883715e+00f) {  /* LOG */
                    return 8.49971399e-02f;
                } else {
                    return -6.81874249e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_21(const float *x) {
    if (x[12] < -8.05117339e-02f) {  /* SpectralEntropy */
        if (x[7] < -3.51383947e-02f) {  /* TFR_MeanEnergy */
            if (x[6] < -3.62451226e-02f) {  /* SSI */
                if (x[12] < -9.93368179e-02f) {  /* SpectralEntropy */
                    return 1.13576710e-01f;
                } else {
                    return -2.28267889e-02f;
                }
            } else {
                if (x[12] < -9.49216411e-02f) {  /* SpectralEntropy */
                    return -8.24560598e-03f;
                } else {
                    return 3.01758349e-02f;
                }
            }
        } else {
            if (x[12] < -9.87810418e-02f) {  /* SpectralEntropy */
                if (x[2] < -9.95433569e-01f) {  /* WAMP */
                    return -2.63954308e-02f;
                } else {
                    return -1.08626544e-01f;
                }
            } else {
                if (x[14] < -4.97429892e-02f) {  /* MeanFreq */
                    return -2.33689398e-02f;
                } else {
                    return -6.96561113e-02f;
                }
            }
        }
    } else {
        if (x[2] < 1.99234867e+00f) {  /* WAMP */
            if (x[0] < -3.52923051e-02f) {  /* MAV */
                if (x[6] < -3.62462178e-02f) {  /* SSI */
                    return -3.42434435e-03f;
                } else {
                    return 4.04999815e-02f;
                }
            } else {
                if (x[0] < 4.67124060e-02f) {  /* MAV */
                    return -1.19475427e-03f;
                } else {
                    return -1.07228078e-01f;
                }
            }
        } else {
            if (x[1] < -3.54885310e-02f) {  /* WL */
                return -9.74814743e-02f;
            } else {
                if (x[7] < -3.51309516e-02f) {  /* TFR_MeanEnergy */
                    return 9.07075331e-02f;
                } else {
                    return 3.17601971e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_22(const float *x) {
    if (x[10] < -4.66369651e-02f) {  /* R_Mid */
        if (x[7] < -3.51350978e-02f) {  /* TFR_MeanEnergy */
            if (x[7] < -3.51462141e-02f) {  /* TFR_MeanEnergy */
                if (x[0] < -1.66312024e-01f) {  /* MAV */
                    return 7.60259256e-02f;
                } else {
                    return -5.51897660e-02f;
                }
            } else {
                if (x[12] < -8.87201652e-02f) {  /* SpectralEntropy */
                    return -1.03850774e-02f;
                } else {
                    return 1.38346376e-02f;
                }
            }
        } else {
            if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                if (x[1] < -3.72548439e-02f) {  /* WL */
                    return -6.89047351e-02f;
                } else {
                    return 1.64640823e-03f;
                }
            } else {
                if (x[3] < -3.62440236e-02f) {  /* VAR */
                    return -7.31550455e-02f;
                } else {
                    return -8.73852707e-03f;
                }
            }
        }
    } else {
        if (x[1] < -3.53484228e-02f) {  /* WL */
            if (x[14] < -4.20331433e-02f) {  /* MeanFreq */
                if (x[2] < 1.59219909e+00f) {  /* WAMP */
                    return 3.79053727e-02f;
                } else {
                    return -6.94522960e-03f;
                }
            } else {
                if (x[14] < -5.23453020e-03f) {  /* MeanFreq */
                    return -5.03414050e-02f;
                } else {
                    return 9.89706293e-02f;
                }
            }
        } else {
            if (x[2] < -5.68607569e-01f) {  /* WAMP */
                return -9.42088515e-02f;
            } else {
                if (x[2] < 1.75225902e+00f) {  /* WAMP */
                    return 3.54974009e-02f;
                } else {
                    return 8.00113380e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_23(const float *x) {
    if (x[13] < -6.09963298e-01f) {  /* SpectralKurtosis */
        if (x[9] < -3.61588970e-02f) {  /* E_Low */
            if (x[10] < -4.97573726e-02f) {  /* R_Mid */
                if (x[12] < 1.65871251e-02f) {  /* SpectralEntropy */
                    return 1.58295091e-02f;
                } else {
                    return -1.00008838e-01f;
                }
            } else {
                if (x[6] < -3.62484343e-02f) {  /* SSI */
                    return 8.64297431e-03f;
                } else {
                    return 6.23487718e-02f;
                }
            }
        } else {
            if (x[2] < -2.48488054e-01f) {  /* WAMP */
                return -1.08719014e-01f;
            } else {
                if (x[10] < -5.00260070e-02f) {  /* R_Mid */
                    return -1.20117545e-01f;
                } else {
                    return 4.67521511e-03f;
                }
            }
        }
    } else {
        if (x[7] < -3.51380780e-02f) {  /* TFR_MeanEnergy */
            if (x[2] < 3.11721087e-01f) {  /* WAMP */
                if (x[7] < -3.51398736e-02f) {  /* TFR_MeanEnergy */
                    return -1.64731871e-02f;
                } else {
                    return 1.00431675e-02f;
                }
            } else {
                if (x[8] < -3.62437181e-02f) {  /* E_Total */
                    return 3.64835896e-02f;
                } else {
                    return -1.68957740e-01f;
                }
            }
        } else {
            if (x[12] < -7.31797218e-02f) {  /* SpectralEntropy */
                if (x[12] < -9.86079052e-02f) {  /* SpectralEntropy */
                    return -1.02275267e-01f;
                } else {
                    return -3.13277245e-02f;
                }
            } else {
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return 1.94307901e-02f;
                } else {
                    return -2.91419607e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_24(const float *x) {
    if (x[10] < -4.76952605e-02f) {  /* R_Mid */
        if (x[7] < -3.51371579e-02f) {  /* TFR_MeanEnergy */
            if (x[7] < -3.51402126e-02f) {  /* TFR_MeanEnergy */
                if (x[0] < -1.66312024e-01f) {  /* MAV */
                    return 1.12068467e-01f;
                } else {
                    return -1.42719718e-02f;
                }
            } else {
                if (x[10] < -5.05131334e-02f) {  /* R_Mid */
                    return -4.51620156e-03f;
                } else {
                    return 2.17745174e-02f;
                }
            }
        } else {
            if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                if (x[10] < -4.94689420e-02f) {  /* R_Mid */
                    return -2.25902591e-02f;
                } else {
                    return 2.24380773e-02f;
                }
            } else {
                if (x[3] < -3.62440348e-02f) {  /* VAR */
                    return -6.13287985e-02f;
                } else {
                    return -6.68836618e-03f;
                }
            }
        }
    } else {
        if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
            if (x[0] < -1.26246616e-01f) {  /* MAV */
                if (x[10] < -4.66369651e-02f) {  /* R_Mid */
                    return -4.51163873e-02f;
                } else {
                    return 8.41238573e-02f;
                }
            } else {
                if (x[1] < -3.65998745e-02f) {  /* WL */
                    return -7.23796263e-02f;
                } else {
                    return 7.22071854e-03f;
                }
            }
        } else {
            if (x[13] < -6.20749593e-01f) {  /* SpectralKurtosis */
                if (x[2] < 9.83080864e-02f) {  /* WAMP */
                    return -7.85613954e-02f;
                } else {
                    return 7.48378262e-02f;
                }
            } else {
                if (x[2] < 1.48549259e+00f) {  /* WAMP */
                    return 3.47254612e-02f;
                } else {
                    return -5.52455755e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_25(const float *x) {
    if (x[12] < -7.12123960e-02f) {  /* SpectralEntropy */
        if (x[2] < 6.85193837e-01f) {  /* WAMP */
            if (x[12] < -9.59550962e-02f) {  /* SpectralEntropy */
                if (x[2] < -8.08697224e-01f) {  /* WAMP */
                    return -1.13358004e-02f;
                } else {
                    return -6.62840605e-02f;
                }
            } else {
                if (x[5] < -9.54879820e-01f) {  /* LOG */
                    return -4.82329950e-02f;
                } else {
                    return 1.13566068e-03f;
                }
            }
        } else {
            return -1.20027721e-01f;
        }
    } else {
        if (x[13] < -6.20130956e-01f) {  /* SpectralKurtosis */
            if (x[8] < -3.62276025e-02f) {  /* E_Total */
                if (x[1] < -3.61033753e-02f) {  /* WL */
                    return -1.42138898e-02f;
                } else {
                    return 6.19710349e-02f;
                }
            } else {
                if (x[2] < -5.41930914e-01f) {  /* WAMP */
                    return -1.01639472e-01f;
                } else {
                    return 2.07252870e-03f;
                }
            }
        } else {
            if (x[4] < -1.34111738e+00f) {  /* SNR */
                if (x[1] < -3.50926928e-02f) {  /* WL */
                    return -7.35705718e-02f;
                } else {
                    return 4.53966707e-02f;
                }
            } else {
                if (x[5] < -7.97443569e-01f) {  /* LOG */
                    return -1.87378954e-02f;
                } else {
                    return 1.79051142e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_26(const float *x) {
    if (x[10] < -4.66369651e-02f) {  /* R_Mid */
        if (x[5] < -2.03995481e-02f) {  /* LOG */
            if (x[6] < -3.62451226e-02f) {  /* SSI */
                if (x[13] < 3.63706279e+00f) {  /* SpectralKurtosis */
                    return -9.48795211e-03f;
                } else {
                    return 1.62709877e-01f;
                }
            } else {
                if (x[10] < -5.04248068e-02f) {  /* R_Mid */
                    return 9.98921786e-03f;
                } else {
                    return 4.59094755e-02f;
                }
            }
        } else {
            if (x[12] < -9.87810418e-02f) {  /* SpectralEntropy */
                if (x[7] < -3.51387784e-02f) {  /* TFR_MeanEnergy */
                    return -3.80306318e-02f;
                } else {
                    return -1.03721976e-01f;
                }
            } else {
                if (x[4] < 7.96598792e-01f) {  /* SNR */
                    return -1.02045340e-02f;
                } else {
                    return -3.27321701e-02f;
                }
            }
        }
    } else {
        if (x[5] < -4.81943178e+00f) {  /* LOG */
            return 1.24560237e-01f;
        } else {
            if (x[7] < -3.51429246e-02f) {  /* TFR_MeanEnergy */
                if (x[10] < -4.45837751e-02f) {  /* R_Mid */
                    return 2.65486520e-02f;
                } else {
                    return -3.24458778e-02f;
                }
            } else {
                if (x[7] < -3.51318978e-02f) {  /* TFR_MeanEnergy */
                    return 4.01932113e-02f;
                } else {
                    return 1.00331474e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_27(const float *x) {
    if (x[10] < -4.74665686e-02f) {  /* R_Mid */
        if (x[5] < 4.50353891e-01f) {  /* LOG */
            if (x[13] < -3.59944671e-01f) {  /* SpectralKurtosis */
                if (x[6] < -3.62486690e-02f) {  /* SSI */
                    return -2.52977964e-02f;
                } else {
                    return 1.06864069e-02f;
                }
            } else {
                if (x[6] < -3.62452790e-02f) {  /* SSI */
                    return -2.08972022e-02f;
                } else {
                    return -2.46626604e-03f;
                }
            }
        } else {
            if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                if (x[3] < -3.62437218e-02f) {  /* VAR */
                    return -7.77317537e-03f;
                } else {
                    return -1.08563580e-01f;
                }
            } else {
                if (x[3] < -3.62440273e-02f) {  /* VAR */
                    return -7.41757900e-02f;
                } else {
                    return 5.44883590e-03f;
                }
            }
        }
    } else {
        if (x[5] < 7.21219361e-01f) {  /* LOG */
            if (x[2] < 2.12573171e+00f) {  /* WAMP */
                if (x[5] < -6.06637061e-01f) {  /* LOG */
                    return -2.29925453e-03f;
                } else {
                    return 3.22979987e-02f;
                }
            } else {
                if (x[7] < -3.51306610e-02f) {  /* TFR_MeanEnergy */
                    return 8.87597129e-02f;
                } else {
                    return -2.83950772e-02f;
                }
            }
        } else {
            if (x[2] < -1.15104921e-01f) {  /* WAMP */
                return -1.17799714e-01f;
            } else {
                if (x[13] < -6.21788561e-01f) {  /* SpectralKurtosis */
                    return 4.43659797e-02f;
                } else {
                    return -2.98691075e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_28(const float *x) {
    if (x[1] < -3.59068140e-02f) {  /* WL */
        if (x[0] < 4.67124060e-02f) {  /* MAV */
            if (x[5] < -4.81943178e+00f) {  /* LOG */
                if (x[3] < -3.62441167e-02f) {  /* VAR */
                    return 1.79573402e-01f;
                } else {
                    return 4.69265580e-02f;
                }
            } else {
                if (x[5] < -8.97694528e-01f) {  /* LOG */
                    return -3.95690165e-02f;
                } else {
                    return -2.06955406e-03f;
                }
            }
        } else {
            return -1.00654282e-01f;
        }
    } else {
        if (x[7] < -3.51297967e-02f) {  /* TFR_MeanEnergy */
            if (x[3] < -3.62439789e-02f) {  /* VAR */
                if (x[0] < -2.16653459e-02f) {  /* MAV */
                    return -5.03869615e-02f;
                } else {
                    return 6.58960044e-02f;
                }
            } else {
                if (x[4] < -1.27924943e+00f) {  /* SNR */
                    return 2.96374355e-02f;
                } else {
                    return 6.36008978e-02f;
                }
            }
        } else {
            if (x[4] < -1.57363224e+00f) {  /* SNR */
                if (x[1] < -3.45213786e-02f) {  /* WL */
                    return -1.40123084e-01f;
                } else {
                    return -5.45992795e-03f;
                }
            } else {
                if (x[1] < -3.50219570e-02f) {  /* WL */
                    return -1.71832610e-02f;
                } else {
                    return 9.70585346e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_29(const float *x) {
    if (x[10] < -4.99075763e-02f) {  /* R_Mid */
        if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
            if (x[4] < 7.96598792e-01f) {  /* SNR */
                if (x[13] < 1.33178845e-01f) {  /* SpectralKurtosis */
                    return 5.42477518e-03f;
                } else {
                    return 3.45995389e-02f;
                }
            } else {
                if (x[0] < -3.79430652e-02f) {  /* MAV */
                    return 1.53764745e-03f;
                } else {
                    return -5.58016486e-02f;
                }
            }
        } else {
            if (x[8] < -3.62375751e-02f) {  /* E_Total */
                if (x[13] < 1.73085487e+00f) {  /* SpectralKurtosis */
                    return -1.71552151e-02f;
                } else {
                    return 1.83919203e-02f;
                }
            } else {
                if (x[5] < 4.83148384e+00f) {  /* LOG */
                    return -4.56688739e-02f;
                } else {
                    return -9.72661674e-02f;
                }
            }
        }
    } else {
        if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
            if (x[5] < -4.81943178e+00f) {  /* LOG */
                if (x[2] < -9.95433569e-01f) {  /* WAMP */
                    return -4.47222497e-03f;
                } else {
                    return 8.11145529e-02f;
                }
            } else {
                if (x[1] < -3.72249186e-02f) {  /* WL */
                    return -9.67149362e-02f;
                } else {
                    return -1.48008754e-02f;
                }
            }
        } else {
            if (x[8] < -3.62528414e-02f) {  /* E_Total */
                if (x[13] < -6.18971348e-01f) {  /* SpectralKurtosis */
                    return 4.49162535e-02f;
                } else {
                    return -3.24900262e-02f;
                }
            } else {
                if (x[5] < -2.23260112e-02f) {  /* LOG */
                    return 3.39009203e-02f;
                } else {
                    return 7.06523051e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_30(const float *x) {
    if (x[1] < -3.56551632e-02f) {  /* WL */
        if (x[1] < -3.73126566e-02f) {  /* WL */
            if (x[8] < -3.62448618e-02f) {  /* E_Total */
                if (x[13] < 8.70487034e-01f) {  /* SpectralKurtosis */
                    return -2.94140559e-02f;
                } else {
                    return 6.47565816e-03f;
                }
            } else {
                if (x[4] < 1.62340498e+00f) {  /* SNR */
                    return -4.35904004e-02f;
                } else {
                    return -1.05090618e-01f;
                }
            }
        } else {
            if (x[2] < 1.53884590e+00f) {  /* WAMP */
                if (x[13] < 4.00187463e-01f) {  /* SpectralKurtosis */
                    return 1.66945835e-03f;
                } else {
                    return 5.04313409e-02f;
                }
            } else {
                if (x[3] < -3.62439789e-02f) {  /* VAR */
                    return -8.13458562e-02f;
                } else {
                    return 3.60980420e-03f;
                }
            }
        }
    } else {
        if (x[2] < 1.69890559e+00f) {  /* WAMP */
            if (x[4] < -1.27924943e+00f) {  /* SNR */
                if (x[2] < 1.56552255e+00f) {  /* WAMP */
                    return 6.37588277e-03f;
                } else {
                    return -9.54721794e-02f;
                }
            } else {
                if (x[13] < -6.09963298e-01f) {  /* SpectralKurtosis */
                    return 8.22506174e-02f;
                } else {
                    return -1.09195337e-03f;
                }
            }
        } else {
            if (x[13] < -6.07164919e-01f) {  /* SpectralKurtosis */
                if (x[10] < -4.94689420e-02f) {  /* R_Mid */
                    return -7.14726150e-02f;
                } else {
                    return 6.08698018e-02f;
                }
            } else {
                if (x[4] < -9.05658960e-01f) {  /* SNR */
                    return 2.21963432e-02f;
                } else {
                    return -1.52104378e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_31(const float *x) {
    if (x[5] < -4.81943178e+00f) {  /* LOG */
        if (x[14] < -4.94681522e-02f) {  /* MeanFreq */
            if (x[13] < -3.29933584e-01f) {  /* SpectralKurtosis */
                return 1.88186154e-01f;
            } else {
                return 3.11340690e-02f;
            }
        } else {
            if (x[10] < -4.78219502e-02f) {  /* R_Mid */
                return -7.64560550e-02f;
            } else {
                if (x[4] < -5.41200519e-01f) {  /* SNR */
                    return 9.86357629e-02f;
                } else {
                    return -8.00860859e-03f;
                }
            }
        }
    } else {
        if (x[1] < -3.50219570e-02f) {  /* WL */
            if (x[5] < 1.68236624e-02f) {  /* LOG */
                if (x[6] < -3.62452790e-02f) {  /* SSI */
                    return -6.36099745e-03f;
                } else {
                    return 1.30397305e-02f;
                }
            } else {
                if (x[1] < -3.72915752e-02f) {  /* WL */
                    return -4.68396805e-02f;
                } else {
                    return -7.54487887e-03f;
                }
            }
        } else {
            if (x[2] < 1.69890559e+00f) {  /* WAMP */
                if (x[5] < 3.88727367e-01f) {  /* LOG */
                    return -1.13183677e-01f;
                } else {
                    return 1.07113114e-02f;
                }
            } else {
                if (x[6] < -3.62331904e-02f) {  /* SSI */
                    return 6.79332539e-02f;
                } else {
                    return -3.22281010e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_32(const float *x) {
    if (x[10] < -5.03502190e-02f) {  /* R_Mid */
        if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
            if (x[2] < -8.39841925e-03f) {  /* WAMP */
                if (x[4] < 9.98444915e-01f) {  /* SNR */
                    return 3.33633609e-02f;
                } else {
                    return -8.67176708e-03f;
                }
            } else {
                return -1.56769559e-01f;
            }
        } else {
            if (x[4] < -4.83984262e-01f) {  /* SNR */
                if (x[10] < -5.04499413e-02f) {  /* R_Mid */
                    return 1.10788301e-01f;
                } else {
                    return -2.83361133e-02f;
                }
            } else {
                if (x[5] < -9.54879820e-01f) {  /* LOG */
                    return -8.46141204e-02f;
                } else {
                    return -1.21358139e-02f;
                }
            }
        }
    } else {
        if (x[7] < -3.51381525e-02f) {  /* TFR_MeanEnergy */
            if (x[5] < -9.26018134e-02f) {  /* LOG */
                if (x[5] < -4.81943178e+00f) {  /* LOG */
                    return 8.63658264e-02f;
                } else {
                    return 7.47522060e-03f;
                }
            } else {
                if (x[9] < -3.62568013e-02f) {  /* E_Low */
                    return 6.56262711e-02f;
                } else {
                    return 3.00438982e-02f;
                }
            }
        } else {
            if (x[13] < -5.90602279e-01f) {  /* SpectralKurtosis */
                if (x[9] < -3.61676104e-02f) {  /* E_Low */
                    return 3.29897404e-02f;
                } else {
                    return -1.53551353e-02f;
                }
            } else {
                if (x[2] < 8.18576992e-01f) {  /* WAMP */
                    return -4.23218217e-03f;
                } else {
                    return -4.88680825e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_33(const float *x) {
    if (x[12] < -9.59550962e-02f) {  /* SpectralEntropy */
        if (x[13] < 3.82691652e-01f) {  /* SpectralKurtosis */
            if (x[13] < -4.79845107e-02f) {  /* SpectralKurtosis */
                if (x[1] < -3.72174382e-02f) {  /* WL */
                    return 1.10906340e-01f;
                } else {
                    return -5.76960631e-02f;
                }
            } else {
                if (x[0] < -4.35207896e-02f) {  /* MAV */
                    return -5.18816931e-04f;
                } else {
                    return -8.53138342e-02f;
                }
            }
        } else {
            if (x[6] < -3.62444930e-02f) {  /* SSI */
                if (x[14] < -5.04878275e-02f) {  /* MeanFreq */
                    return 2.39890721e-02f;
                } else {
                    return -1.14951208e-02f;
                }
            } else {
                if (x[12] < -9.87810418e-02f) {  /* SpectralEntropy */
                    return -1.01648390e-01f;
                } else {
                    return -1.95345320e-02f;
                }
            }
        }
    } else {
        if (x[0] < -2.91209184e-02f) {  /* MAV */
            if (x[1] < -3.68358828e-02f) {  /* WL */
                if (x[3] < -3.62440459e-02f) {  /* VAR */
                    return 1.56640157e-03f;
                } else {
                    return -9.29116532e-02f;
                }
            } else {
                if (x[1] < -3.46165970e-02f) {  /* WL */
                    return 1.61799677e-02f;
                } else {
                    return 7.58570656e-02f;
                }
            }
        } else {
            if (x[12] < -5.81776053e-02f) {  /* SpectralEntropy */
                if (x[1] < -3.65611054e-02f) {  /* WL */
                    return -9.14744195e-03f;
                } else {
                    return -6.69549108e-02f;
                }
            } else {
                if (x[4] < -5.32351971e-01f) {  /* SNR */
                    return -4.94941254e-04f;
                } else {
                    return 1.41258702e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_34(const float *x) {
    if (x[10] < -4.66369651e-02f) {  /* R_Mid */
        if (x[5] < 4.01322663e-01f) {  /* LOG */
            if (x[7] < -3.51445340e-02f) {  /* TFR_MeanEnergy */
                if (x[5] < -4.81943178e+00f) {  /* LOG */
                    return 9.79837403e-02f;
                } else {
                    return -3.06597650e-02f;
                }
            } else {
                if (x[10] < -5.06735072e-02f) {  /* R_Mid */
                    return -9.03286971e-03f;
                } else {
                    return 7.39424070e-03f;
                }
            }
        } else {
            if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                if (x[2] < -8.62050474e-01f) {  /* WAMP */
                    return -6.81787506e-02f;
                } else {
                    return -2.00902252e-03f;
                }
            } else {
                if (x[12] < -4.76557165e-02f) {  /* SpectralEntropy */
                    return -5.93512356e-02f;
                } else {
                    return -7.13207712e-03f;
                }
            }
        }
    } else {
        if (x[1] < -3.64121534e-02f) {  /* WL */
            if (x[10] < -4.50112633e-02f) {  /* R_Mid */
                if (x[1] < -3.70521657e-02f) {  /* WL */
                    return -8.12435150e-02f;
                } else {
                    return 4.35772128e-02f;
                }
            } else {
                if (x[5] < -2.47973824e+00f) {  /* LOG */
                    return 6.29308224e-02f;
                } else {
                    return -5.25359102e-02f;
                }
            }
        } else {
            if (x[12] < -5.30751087e-02f) {  /* SpectralEntropy */
                if (x[0] < -1.30397798e-02f) {  /* MAV */
                    return -5.76264933e-02f;
                } else {
                    return 1.23859666e-01f;
                }
            } else {
                if (x[7] < -3.51309516e-02f) {  /* TFR_MeanEnergy */
                    return 3.88538800e-02f;
                } else {
                    return -3.72124254e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_35(const float *x) {
    if (x[10] < -4.98902127e-02f) {  /* R_Mid */
        if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
            if (x[8] < -3.62481177e-02f) {  /* E_Total */
                if (x[0] < -1.66312024e-01f) {  /* MAV */
                    return 1.32857889e-01f;
                } else {
                    return -1.84446275e-02f;
                }
            } else {
                if (x[4] < 8.48652959e-01f) {  /* SNR */
                    return 1.78657155e-02f;
                } else {
                    return -6.34805858e-03f;
                }
            }
        } else {
            if (x[2] < -9.95433569e-01f) {  /* WAMP */
                if (x[0] < -3.47743742e-02f) {  /* MAV */
                    return 3.54057662e-02f;
                } else {
                    return -9.04221535e-02f;
                }
            } else {
                if (x[7] < -3.51394489e-02f) {  /* TFR_MeanEnergy */
                    return -2.68260427e-02f;
                } else {
                    return -9.05456673e-03f;
                }
            }
        }
    } else {
        if (x[9] < -3.62969525e-02f) {  /* E_Low */
            if (x[9] < -3.63977104e-02f) {  /* E_Low */
                if (x[10] < -4.81044054e-02f) {  /* R_Mid */
                    return 5.40001094e-02f;
                } else {
                    return -5.93773369e-03f;
                }
            } else {
                if (x[0] < -7.12766647e-02f) {  /* MAV */
                    return -6.10622056e-02f;
                } else {
                    return -1.09137706e-02f;
                }
            }
        } else {
            if (x[7] < -3.51381525e-02f) {  /* TFR_MeanEnergy */
                if (x[3] < -3.62439118e-02f) {  /* VAR */
                    return 2.80519463e-02f;
                } else {
                    return -1.50069986e-02f;
                }
            } else {
                if (x[1] < -3.53919528e-02f) {  /* WL */
                    return -2.20321608e-03f;
                } else {
                    return 4.36383560e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_36(const float *x) {
    if (x[13] < -6.08117580e-01f) {  /* SpectralKurtosis */
        if (x[10] < -4.88750152e-02f) {  /* R_Mid */
            if (x[12] < 1.95077602e-02f) {  /* SpectralEntropy */
                if (x[10] < -4.92600910e-02f) {  /* R_Mid */
                    return 4.05697189e-02f;
                } else {
                    return -4.97043394e-02f;
                }
            } else {
                if (x[0] < -5.21450303e-02f) {  /* MAV */
                    return 4.15522456e-02f;
                } else {
                    return -8.49693641e-02f;
                }
            }
        } else {
            if (x[14] < -4.54153642e-02f) {  /* MeanFreq */
                if (x[2] < 1.75225902e+00f) {  /* WAMP */
                    return 9.90777686e-02f;
                } else {
                    return -2.07239506e-03f;
                }
            } else {
                if (x[2] < 1.03198993e+00f) {  /* WAMP */
                    return -2.30855569e-02f;
                } else {
                    return 3.06859221e-02f;
                }
            }
        }
    } else {
        if (x[5] < 2.17295122e+00f) {  /* LOG */
            if (x[2] < 1.59219909e+00f) {  /* WAMP */
                if (x[5] < 1.58213675e+00f) {  /* LOG */
                    return -1.05670595e-03f;
                } else {
                    return 1.38882339e-01f;
                }
            } else {
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return 3.54903191e-02f;
                } else {
                    return -7.73028657e-02f;
                }
            }
        } else {
            if (x[3] < -3.62438634e-02f) {  /* VAR */
                return -8.89297426e-02f;
            } else {
                return -1.63350701e-02f;
            }
        }
    }
}

static inline float xgb_tree_37(const float *x) {
    if (x[10] < -4.98902127e-02f) {  /* R_Mid */
        if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
            if (x[13] < 8.14111456e-02f) {  /* SpectralKurtosis */
                if (x[14] < -4.90774028e-02f) {  /* MeanFreq */
                    return -7.58688990e-03f;
                } else {
                    return -1.10536575e-01f;
                }
            } else {
                if (x[14] < -5.04163578e-02f) {  /* MeanFreq */
                    return 6.97487080e-03f;
                } else {
                    return 6.25370517e-02f;
                }
            }
        } else {
            if (x[7] < -3.51371579e-02f) {  /* TFR_MeanEnergy */
                if (x[8] < -3.62468809e-02f) {  /* E_Total */
                    return -1.76667832e-02f;
                } else {
                    return -6.13176089e-04f;
                }
            } else {
                if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                    return -2.52228417e-02f;
                } else {
                    return -8.97470638e-02f;
                }
            }
        }
    } else {
        if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
            if (x[13] < -6.27466917e-01f) {  /* SpectralKurtosis */
                if (x[0] < 4.67124060e-02f) {  /* MAV */
                    return 1.05864622e-01f;
                } else {
                    return -8.34130682e-03f;
                }
            } else {
                if (x[10] < -4.98727895e-02f) {  /* R_Mid */
                    return 1.39235333e-01f;
                } else {
                    return -2.14810856e-02f;
                }
            }
        } else {
            if (x[5] < 3.74677360e-01f) {  /* LOG */
                if (x[0] < -6.10601231e-02f) {  /* MAV */
                    return -1.47642288e-02f;
                } else {
                    return 2.00820100e-02f;
                }
            } else {
                if (x[12] < -4.96421419e-02f) {  /* SpectralEntropy */
                    return -2.69658770e-02f;
                } else {
                    return 1.29039986e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_38(const float *x) {
    if (x[5] < -4.81943178e+00f) {  /* LOG */
        if (x[1] < -3.66468020e-02f) {  /* WL */
            if (x[1] < -3.74854095e-02f) {  /* WL */
                if (x[10] < -4.66369651e-02f) {  /* R_Mid */
                    return -2.04953663e-02f;
                } else {
                    return 5.68374470e-02f;
                }
            } else {
                if (x[2] < -6.75314069e-01f) {  /* WAMP */
                    return 1.46978781e-01f;
                } else {
                    return 5.78479469e-02f;
                }
            }
        } else {
            if (x[2] < -4.08547819e-01f) {  /* WAMP */
                return 3.43833789e-02f;
            } else {
                return -6.49254695e-02f;
            }
        }
    } else {
        if (x[0] < 4.67124060e-02f) {  /* MAV */
            if (x[1] < -3.71732302e-02f) {  /* WL */
                if (x[2] < -7.01990664e-01f) {  /* WAMP */
                    return -4.56007943e-03f;
                } else {
                    return -4.62381653e-02f;
                }
            } else {
                if (x[13] < -6.26930475e-01f) {  /* SpectralKurtosis */
                    return 5.26610687e-02f;
                } else {
                    return 3.47567885e-03f;
                }
            }
        } else {
            if (x[5] < 7.59880185e-01f) {  /* LOG */
                if (x[2] < -5.68607569e-01f) {  /* WAMP */
                    return -6.34023026e-02f;
                } else {
                    return 1.06213287e-01f;
                }
            } else {
                if (x[5] < 1.33196819e+00f) {  /* LOG */
                    return -4.67095636e-02f;
                } else {
                    return -1.01078510e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_39(const float *x) {
    if (x[13] < -6.18349969e-01f) {  /* SpectralKurtosis */
        if (x[4] < -1.17399502e+00f) {  /* SNR */
            if (x[2] < 1.91231871e+00f) {  /* WAMP */
                if (x[10] < -4.93563786e-02f) {  /* R_Mid */
                    return -1.05160825e-01f;
                } else {
                    return -1.62411947e-03f;
                }
            } else {
                if (x[7] < -3.51309516e-02f) {  /* TFR_MeanEnergy */
                    return 7.28940219e-02f;
                } else {
                    return -4.17285739e-03f;
                }
            }
        } else {
            if (x[2] < 4.45104212e-01f) {  /* WAMP */
                return -6.24478832e-02f;
            } else {
                if (x[5] < 1.04134607e+00f) {  /* LOG */
                    return 8.99933353e-02f;
                } else {
                    return -1.92813035e-02f;
                }
            }
        }
    } else {
        if (x[2] < 1.45881605e+00f) {  /* WAMP */
            if (x[4] < -9.75628942e-02f) {  /* SNR */
                if (x[7] < -3.51409949e-02f) {  /* TFR_MeanEnergy */
                    return -6.50957972e-03f;
                } else {
                    return 1.30245863e-02f;
                }
            } else {
                if (x[5] < -4.81943178e+00f) {  /* LOG */
                    return 1.08077124e-01f;
                } else {
                    return -5.29549131e-03f;
                }
            }
        } else {
            if (x[4] < -8.92963409e-01f) {  /* SNR */
                if (x[7] < -3.51297967e-02f) {  /* TFR_MeanEnergy */
                    return 1.31016970e-02f;
                } else {
                    return -8.30171257e-02f;
                }
            } else {
                if (x[5] < -6.26587346e-02f) {  /* LOG */
                    return 4.17032838e-02f;
                } else {
                    return -7.46869966e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_40(const float *x) {
    if (x[5] < -1.51421530e-02f) {  /* LOG */
        if (x[5] < -1.04050420e-01f) {  /* LOG */
            if (x[4] < 2.18982935e+00f) {  /* SNR */
                if (x[10] < -5.06981276e-02f) {  /* R_Mid */
                    return -2.00696792e-02f;
                } else {
                    return 4.58300905e-03f;
                }
            } else {
                if (x[1] < -3.74554843e-02f) {  /* WL */
                    return 4.09803465e-02f;
                } else {
                    return 1.68628693e-01f;
                }
            }
        } else {
            if (x[14] < -5.05093522e-02f) {  /* MeanFreq */
                if (x[5] < -6.99923411e-02f) {  /* LOG */
                    return -2.78585707e-03f;
                } else {
                    return -1.15525641e-01f;
                }
            } else {
                if (x[10] < -5.08288853e-02f) {  /* R_Mid */
                    return 9.87768099e-02f;
                } else {
                    return 1.78634487e-02f;
                }
            }
        }
    } else {
        if (x[4] < 1.52368701e+00f) {  /* SNR */
            if (x[2] < -8.88727069e-01f) {  /* WAMP */
                if (x[12] < -9.38127413e-02f) {  /* SpectralEntropy */
                    return -2.15376951e-02f;
                } else {
                    return -7.26201162e-02f;
                }
            } else {
                if (x[13] < 7.74978697e-01f) {  /* SpectralKurtosis */
                    return -2.99299811e-03f;
                } else {
                    return 1.06055818e-01f;
                }
            }
        } else {
            if (x[2] < -9.68756974e-01f) {  /* WAMP */
                if (x[10] < -5.08236587e-02f) {  /* R_Mid */
                    return 2.73734014e-02f;
                } else {
                    return -8.26831907e-02f;
                }
            } else {
                return -1.07215084e-01f;
            }
        }
    }
}

static inline float xgb_tree_41(const float *x) {
    if (x[7] < -3.51381153e-02f) {  /* TFR_MeanEnergy */
        if (x[7] < -3.51466835e-02f) {  /* TFR_MeanEnergy */
            if (x[1] < -3.74514051e-02f) {  /* WL */
                if (x[10] < -4.85422835e-02f) {  /* R_Mid */
                    return 1.26755744e-01f;
                } else {
                    return 1.92152914e-02f;
                }
            } else {
                if (x[10] < -5.05446196e-02f) {  /* R_Mid */
                    return -9.14062709e-02f;
                } else {
                    return -1.53406458e-02f;
                }
            }
        } else {
            if (x[4] < 3.31234068e-01f) {  /* SNR */
                if (x[6] < -3.62451226e-02f) {  /* SSI */
                    return 5.42171067e-03f;
                } else {
                    return 2.33961809e-02f;
                }
            } else {
                if (x[2] < -7.01990664e-01f) {  /* WAMP */
                    return 2.42398097e-03f;
                } else {
                    return -3.84899005e-02f;
                }
            }
        }
    } else {
        if (x[6] < -3.62442546e-02f) {  /* SSI */
            if (x[10] < -4.68342118e-02f) {  /* R_Mid */
                if (x[7] < -3.51380371e-02f) {  /* TFR_MeanEnergy */
                    return -2.30408311e-02f;
                } else {
                    return -2.54416585e-01f;
                }
            } else {
                if (x[6] < -3.62443253e-02f) {  /* SSI */
                    return -4.96975854e-02f;
                } else {
                    return 7.39938691e-02f;
                }
            }
        } else {
            if (x[10] < -4.66369651e-02f) {  /* R_Mid */
                if (x[2] < 6.31840587e-01f) {  /* WAMP */
                    return -6.82180142e-03f;
                } else {
                    return -3.08134537e-02f;
                }
            } else {
                if (x[7] < -3.51316892e-02f) {  /* TFR_MeanEnergy */
                    return 2.44086366e-02f;
                } else {
                    return -1.20501639e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_42(const float *x) {
    if (x[0] < 4.67124060e-02f) {  /* MAV */
        if (x[1] < -3.50926928e-02f) {  /* WL */
            if (x[4] < -1.34111738e+00f) {  /* SNR */
                if (x[14] < -5.23453020e-03f) {  /* MeanFreq */
                    return -6.42148331e-02f;
                } else {
                    return 9.57903787e-02f;
                }
            } else {
                if (x[13] < -6.07164919e-01f) {  /* SpectralKurtosis */
                    return 2.06902903e-02f;
                } else {
                    return -2.26968061e-03f;
                }
            }
        } else {
            if (x[7] < -3.51267308e-02f) {  /* TFR_MeanEnergy */
                if (x[1] < -3.42500024e-02f) {  /* WL */
                    return 3.90434824e-02f;
                } else {
                    return 8.53291899e-02f;
                }
            } else {
                if (x[4] < -1.57363224e+00f) {  /* SNR */
                    return -1.08475022e-01f;
                } else {
                    return 4.25355509e-02f;
                }
            }
        }
    } else {
        if (x[5] < 1.33196819e+00f) {  /* LOG */
            if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                if (x[5] < 7.59880185e-01f) {  /* LOG */
                    return 6.14417810e-03f;
                } else {
                    return -7.65893683e-02f;
                }
            } else {
                return 5.49275950e-02f;
            }
        } else {
            if (x[4] < 1.78322479e-01f) {  /* SNR */
                return -1.14201538e-01f;
            } else {
                if (x[1] < -3.72487232e-02f) {  /* WL */
                    return -8.78257081e-02f;
                } else {
                    return 8.87692794e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_43(const float *x) {
    if (x[0] < 4.67124060e-02f) {  /* MAV */
        if (x[10] < -4.98727895e-02f) {  /* R_Mid */
            if (x[12] < 4.35201451e-02f) {  /* SpectralEntropy */
                if (x[7] < -3.51399146e-02f) {  /* TFR_MeanEnergy */
                    return -1.29257310e-02f;
                } else {
                    return -3.57491626e-05f;
                }
            } else {
                if (x[1] < -3.53919528e-02f) {  /* WL */
                    return -1.89520165e-01f;
                } else {
                    return -5.57984374e-02f;
                }
            }
        } else {
            if (x[13] < -6.26930475e-01f) {  /* SpectralKurtosis */
                if (x[3] < -3.62431556e-02f) {  /* VAR */
                    return 2.35672593e-02f;
                } else {
                    return 8.88242871e-02f;
                }
            } else {
                if (x[3] < -3.62434611e-02f) {  /* VAR */
                    return 5.59541723e-03f;
                } else {
                    return -8.31160173e-02f;
                }
            }
        }
    } else {
        if (x[5] < 8.80530715e-01f) {  /* LOG */
            if (x[5] < 4.26422298e-01f) {  /* LOG */
                return -1.42371077e-02f;
            } else {
                return 3.77129354e-02f;
            }
        } else {
            if (x[4] < 3.95454675e-01f) {  /* SNR */
                return -1.07817553e-01f;
            } else {
                if (x[9] < -3.57143469e-02f) {  /* E_Low */
                    return 4.50455509e-02f;
                } else {
                    return -8.42725560e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_44(const float *x) {
    if (x[6] < -3.62486690e-02f) {  /* SSI */
        if (x[5] < -4.81943178e+00f) {  /* LOG */
            if (x[3] < -3.62441279e-02f) {  /* VAR */
                return 1.37819946e-01f;
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return -3.74487042e-03f;
                } else {
                    return 8.24586153e-02f;
                }
            }
        } else {
            if (x[12] < -7.98559561e-02f) {  /* SpectralEntropy */
                if (x[4] < 5.92501163e-01f) {  /* SNR */
                    return -7.92866275e-02f;
                } else {
                    return -9.28481109e-03f;
                }
            } else {
                if (x[10] < -5.04331663e-02f) {  /* R_Mid */
                    return 1.38057694e-01f;
                } else {
                    return -1.91738568e-02f;
                }
            }
        }
    } else {
        if (x[12] < -9.60454792e-02f) {  /* SpectralEntropy */
            if (x[6] < -3.62446420e-02f) {  /* SSI */
                if (x[14] < -5.04292250e-02f) {  /* MeanFreq */
                    return 9.73347388e-03f;
                } else {
                    return -3.13086882e-02f;
                }
            } else {
                if (x[1] < -3.73126566e-02f) {  /* WL */
                    return -4.08814959e-02f;
                } else {
                    return 1.21918181e-02f;
                }
            }
        } else {
            if (x[11] < 5.80264390e-01f) {  /* TopFreq2 */
                if (x[7] < -3.51383947e-02f) {  /* TFR_MeanEnergy */
                    return 8.50785151e-03f;
                } else {
                    return -3.38611449e-03f;
                }
            } else {
                if (x[5] < -2.10715100e-01f) {  /* LOG */
                    return 1.82250701e-03f;
                } else {
                    return 1.16706245e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_45(const float *x) {
    if (x[2] < 2.12573171e+00f) {  /* WAMP */
        if (x[7] < -3.51402126e-02f) {  /* TFR_MeanEnergy */
            if (x[4] < 2.18982935e+00f) {  /* SNR */
                if (x[2] < 5.78487337e-01f) {  /* WAMP */
                    return -1.22974282e-02f;
                } else {
                    return 1.51700154e-02f;
                }
            } else {
                if (x[1] < -3.74201164e-02f) {  /* WL */
                    return -1.52655202e-03f;
                } else {
                    return 1.93131551e-01f;
                }
            }
        } else {
            if (x[6] < -3.62445712e-02f) {  /* SSI */
                if (x[10] < -5.06981276e-02f) {  /* R_Mid */
                    return -2.34047021e-03f;
                } else {
                    return 1.85716171e-02f;
                }
            } else {
                if (x[12] < -9.87810418e-02f) {  /* SpectralEntropy */
                    return -9.18756947e-02f;
                } else {
                    return -4.50658752e-03f;
                }
            }
        }
    } else {
        if (x[7] < -3.51306610e-02f) {  /* TFR_MeanEnergy */
            if (x[4] < -1.45617855e+00f) {  /* SNR */
                if (x[13] < -6.30430877e-01f) {  /* SpectralKurtosis */
                    return 4.17225882e-02f;
                } else {
                    return 9.65333804e-02f;
                }
            } else {
                if (x[13] < -6.23944223e-01f) {  /* SpectralKurtosis */
                    return -1.30510837e-01f;
                } else {
                    return 8.24018344e-02f;
                }
            }
        } else {
            if (x[5] < 9.27969217e-01f) {  /* LOG */
                if (x[2] < 2.39249802e+00f) {  /* WAMP */
                    return -1.41729072e-01f;
                } else {
                    return -1.27824908e-02f;
                }
            } else {
                if (x[12] < 1.21707410e-01f) {  /* SpectralEntropy */
                    return 8.36405456e-02f;
                } else {
                    return 2.63748714e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_46(const float *x) {
    if (x[5] < -4.81943178e+00f) {  /* LOG */
        if (x[3] < -3.62440906e-02f) {  /* VAR */
            if (x[13] < -3.95395875e-01f) {  /* SpectralKurtosis */
                return 1.13151789e-01f;
            } else {
                return -5.10700804e-04f;
            }
        } else {
            if (x[10] < -3.31971049e-02f) {  /* R_Mid */
                if (x[3] < -3.62440757e-02f) {  /* VAR */
                    return 1.06022814e-02f;
                } else {
                    return -6.47255108e-02f;
                }
            } else {
                if (x[1] < -3.63727100e-02f) {  /* WL */
                    return 8.15238133e-02f;
                } else {
                    return -3.39980423e-02f;
                }
            }
        }
    } else {
        if (x[7] < -3.51462141e-02f) {  /* TFR_MeanEnergy */
            if (x[1] < -3.74514051e-02f) {  /* WL */
                return 1.56485021e-01f;
            } else {
                if (x[1] < -3.71603072e-02f) {  /* WL */
                    return -5.58040626e-02f;
                } else {
                    return -1.49487704e-02f;
                }
            }
        } else {
            if (x[10] < -4.93846573e-02f) {  /* R_Mid */
                if (x[12] < 1.95077602e-02f) {  /* SpectralEntropy */
                    return -1.96752721e-03f;
                } else {
                    return -8.86380449e-02f;
                }
            } else {
                if (x[1] < -3.71562243e-02f) {  /* WL */
                    return -4.20918055e-02f;
                } else {
                    return 1.05653061e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_47(const float *x) {
    if (x[13] < 8.70487034e-01f) {  /* SpectralKurtosis */
        if (x[4] < 1.19614708e+00f) {  /* SNR */
            if (x[11] < 5.80264390e-01f) {  /* TopFreq2 */
                if (x[10] < -5.07192351e-02f) {  /* R_Mid */
                    return -1.77284125e-02f;
                } else {
                    return 1.55319239e-03f;
                }
            } else {
                if (x[13] < -7.12004676e-02f) {  /* SpectralKurtosis */
                    return 9.40659270e-02f;
                } else {
                    return -6.83844015e-02f;
                }
            }
        } else {
            if (x[13] < -9.07716528e-02f) {  /* SpectralKurtosis */
                return 1.10330477e-01f;
            } else {
                if (x[10] < -5.08119687e-02f) {  /* R_Mid */
                    return 6.56470880e-02f;
                } else {
                    return -7.42686018e-02f;
                }
            }
        }
    } else {
        if (x[6] < -3.62429023e-02f) {  /* SSI */
            if (x[9] < -3.62638645e-02f) {  /* E_Low */
                if (x[4] < 2.37086248e+00f) {  /* SNR */
                    return -7.18053756e-03f;
                } else {
                    return 1.19441807e-01f;
                }
            } else {
                if (x[0] < -3.95021550e-02f) {  /* MAV */
                    return 6.82488307e-02f;
                } else {
                    return 5.43858251e-03f;
                }
            }
        } else {
            return -8.74649659e-02f;
        }
    }
}

static inline float xgb_tree_48(const float *x) {
    if (x[10] < -4.66369651e-02f) {  /* R_Mid */
        if (x[5] < 8.18698853e-02f) {  /* LOG */
            if (x[2] < -9.95433569e-01f) {  /* WAMP */
                if (x[10] < -5.06981276e-02f) {  /* R_Mid */
                    return 4.57710735e-02f;
                } else {
                    return -2.72219926e-02f;
                }
            } else {
                if (x[10] < -5.06858900e-02f) {  /* R_Mid */
                    return -1.03910668e-02f;
                } else {
                    return 3.38057731e-03f;
                }
            }
        } else {
            if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                if (x[10] < -4.94689420e-02f) {  /* R_Mid */
                    return -1.63940825e-02f;
                } else {
                    return 1.12146102e-02f;
                }
            } else {
                if (x[2] < 5.78487337e-01f) {  /* WAMP */
                    return -2.22125999e-03f;
                } else {
                    return -4.91522439e-02f;
                }
            }
        }
    } else {
        if (x[10] < -3.68184820e-02f) {  /* R_Mid */
            if (x[2] < 1.59219909e+00f) {  /* WAMP */
                if (x[1] < -3.59360613e-02f) {  /* WL */
                    return 2.13853233e-02f;
                } else {
                    return 6.67834580e-02f;
                }
            } else {
                if (x[3] < -3.62439826e-02f) {  /* VAR */
                    return -7.84871355e-02f;
                } else {
                    return 1.52942883e-02f;
                }
            }
        } else {
            if (x[2] < 1.69890559e+00f) {  /* WAMP */
                if (x[10] < -2.43523027e-04f) {  /* R_Mid */
                    return -4.55395691e-02f;
                } else {
                    return 2.29368266e-02f;
                }
            } else {
                if (x[6] < -3.62331904e-02f) {  /* SSI */
                    return 3.67363803e-02f;
                } else {
                    return -8.15488473e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_49(const float *x) {
    if (x[1] < -3.73126566e-02f) {  /* WL */
        if (x[13] < 8.70487034e-01f) {  /* SpectralKurtosis */
            if (x[7] < -3.51577438e-02f) {  /* TFR_MeanEnergy */
                if (x[2] < -9.95433569e-01f) {  /* WAMP */
                    return -3.34652774e-02f;
                } else {
                    return 1.11086689e-01f;
                }
            } else {
                if (x[1] < -3.74724865e-02f) {  /* WL */
                    return 1.17818885e-01f;
                } else {
                    return -3.07169408e-02f;
                }
            }
        } else {
            if (x[1] < -3.74309979e-02f) {  /* WL */
                if (x[13] < 2.21051526e+00f) {  /* SpectralKurtosis */
                    return -4.14856374e-02f;
                } else {
                    return 4.46792459e-03f;
                }
            } else {
                if (x[7] < -3.51387560e-02f) {  /* TFR_MeanEnergy */
                    return 3.23845185e-02f;
                } else {
                    return -2.54469570e-02f;
                }
            }
        }
    } else {
        if (x[13] < 4.94195879e-01f) {  /* SpectralKurtosis */
            if (x[12] < -9.68171358e-02f) {  /* SpectralEntropy */
                if (x[13] < -4.79845107e-02f) {  /* SpectralKurtosis */
                    return 1.27631789e-02f;
                } else {
                    return -7.59686381e-02f;
                }
            } else {
                if (x[2] < -8.88727069e-01f) {  /* WAMP */
                    return -2.38694306e-02f;
                } else {
                    return 3.58013040e-03f;
                }
            }
        } else {
            if (x[7] < -3.51393335e-02f) {  /* TFR_MeanEnergy */
                if (x[1] < -3.72942910e-02f) {  /* WL */
                    return -5.33860512e-02f;
                } else {
                    return 6.45544976e-02f;
                }
            } else {
                if (x[2] < -7.28667319e-01f) {  /* WAMP */
                    return 1.40012681e-01f;
                } else {
                    return -6.19623028e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_50(const float *x) {
    if (x[0] < -1.95803400e-02f) {  /* MAV */
        if (x[10] < -4.74665686e-02f) {  /* R_Mid */
            if (x[7] < -3.51386331e-02f) {  /* TFR_MeanEnergy */
                if (x[14] < -4.55845259e-02f) {  /* MeanFreq */
                    return 3.71191953e-03f;
                } else {
                    return -5.98976426e-02f;
                }
            } else {
                if (x[7] < -3.51385809e-02f) {  /* TFR_MeanEnergy */
                    return -5.70857935e-02f;
                } else {
                    return -6.99406443e-03f;
                }
            }
        } else {
            if (x[7] < -3.51349004e-02f) {  /* TFR_MeanEnergy */
                if (x[0] < -2.42815707e-02f) {  /* MAV */
                    return 9.66776628e-03f;
                } else {
                    return -1.30513445e-01f;
                }
            } else {
                if (x[14] < -4.76464033e-02f) {  /* MeanFreq */
                    return 1.07790567e-01f;
                } else {
                    return 3.35811488e-02f;
                }
            }
        }
    } else {
        if (x[5] < 9.27969217e-01f) {  /* LOG */
            if (x[12] < -4.76557165e-02f) {  /* SpectralEntropy */
                if (x[1] < -3.71201783e-02f) {  /* WL */
                    return 1.28565198e-02f;
                } else {
                    return -6.76923543e-02f;
                }
            } else {
                if (x[6] < -3.62393558e-02f) {  /* SSI */
                    return 2.09365990e-02f;
                } else {
                    return -6.08923249e-02f;
                }
            }
        } else {
            if (x[0] < 4.67124060e-02f) {  /* MAV */
                if (x[12] < -4.33001779e-02f) {  /* SpectralEntropy */
                    return 4.08814736e-02f;
                } else {
                    return -9.41334665e-03f;
                }
            } else {
                if (x[1] < -3.56959701e-02f) {  /* WL */
                    return -9.12487134e-02f;
                } else {
                    return -7.56670395e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_51(const float *x) {
    if (x[8] < -3.61802876e-02f) {  /* E_Total */
        if (x[13] < -6.26930475e-01f) {  /* SpectralKurtosis */
            if (x[4] < -2.40263295e+00f) {  /* SNR */
                if (x[3] < -3.62413004e-02f) {  /* VAR */
                    return -6.07483797e-02f;
                } else {
                    return -1.19127901e-02f;
                }
            } else {
                if (x[10] < -4.96049933e-02f) {  /* R_Mid */
                    return -3.41468006e-02f;
                } else {
                    return 7.31178671e-02f;
                }
            }
        } else {
            if (x[12] < 2.81396747e-01f) {  /* SpectralEntropy */
                if (x[4] < 1.82875657e+00f) {  /* SNR */
                    return 1.39058393e-03f;
                } else {
                    return -2.48262715e-02f;
                }
            } else {
                if (x[10] < -1.40441740e-02f) {  /* R_Mid */
                    return -5.42870425e-02f;
                } else {
                    return -1.84894204e-01f;
                }
            }
        }
    } else {
        if (x[3] < -3.62437516e-02f) {  /* VAR */
            return -9.04229656e-02f;
        } else {
            if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                return -5.79813961e-03f;
            } else {
                if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                    return -7.43788108e-02f;
                } else {
                    return -2.01644395e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_52(const float *x) {
    if (x[1] < -3.50926928e-02f) {  /* WL */
        if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
            if (x[8] < -3.62481177e-02f) {  /* E_Total */
                if (x[10] < -5.07096276e-02f) {  /* R_Mid */
                    return 4.76013683e-02f;
                } else {
                    return -1.18981814e-02f;
                }
            } else {
                if (x[0] < -3.75898033e-02f) {  /* MAV */
                    return 1.84603017e-02f;
                } else {
                    return -1.04866480e-03f;
                }
            }
        } else {
            if (x[12] < -9.90521535e-02f) {  /* SpectralEntropy */
                if (x[7] < -3.51414718e-02f) {  /* TFR_MeanEnergy */
                    return 8.01372752e-02f;
                } else {
                    return -5.61476834e-02f;
                }
            } else {
                if (x[10] < -5.08186333e-02f) {  /* R_Mid */
                    return 8.96472931e-02f;
                } else {
                    return -3.06142727e-03f;
                }
            }
        }
    } else {
        if (x[2] < -5.68607569e-01f) {  /* WAMP */
            return -7.76916742e-02f;
        } else {
            if (x[3] < -3.62438224e-02f) {  /* VAR */
                if (x[7] < -3.51366252e-02f) {  /* TFR_MeanEnergy */
                    return -1.45957455e-01f;
                } else {
                    return 3.07063516e-02f;
                }
            } else {
                if (x[7] < -3.51408385e-02f) {  /* TFR_MeanEnergy */
                    return 9.36802849e-02f;
                } else {
                    return 4.30260859e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_53(const float *x) {
    if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
        if (x[2] < -9.42080319e-01f) {  /* WAMP */
            if (x[10] < -4.66369651e-02f) {  /* R_Mid */
                if (x[4] < 4.33944255e-01f) {  /* SNR */
                    return -5.74590676e-02f;
                } else {
                    return -4.57958644e-03f;
                }
            } else {
                if (x[10] < -4.53790575e-02f) {  /* R_Mid */
                    return 3.10256016e-02f;
                } else {
                    return 1.10290848e-01f;
                }
            }
        } else {
            if (x[12] < 8.18656161e-02f) {  /* SpectralEntropy */
                if (x[5] < 5.43280900e-01f) {  /* LOG */
                    return 7.81502947e-03f;
                } else {
                    return 3.36524807e-02f;
                }
            } else {
                if (x[12] < 5.29933393e-01f) {  /* SpectralEntropy */
                    return -5.42356037e-02f;
                } else {
                    return 4.05663699e-02f;
                }
            }
        }
    } else {
        if (x[10] < -5.06777093e-02f) {  /* R_Mid */
            if (x[2] < -9.68756974e-01f) {  /* WAMP */
                if (x[7] < -3.51407565e-02f) {  /* TFR_MeanEnergy */
                    return -5.46672232e-02f;
                } else {
                    return 3.52519937e-02f;
                }
            } else {
                if (x[0] < -3.83866169e-02f) {  /* MAV */
                    return -4.13674563e-02f;
                } else {
                    return -2.30039307e-03f;
                }
            }
        } else {
            if (x[7] < -3.51478942e-02f) {  /* TFR_MeanEnergy */
                if (x[5] < -1.74581444e+00f) {  /* LOG */
                    return 4.15450614e-03f;
                } else {
                    return -6.27934486e-02f;
                }
            } else {
                if (x[1] < -3.71201783e-02f) {  /* WL */
                    return 1.25737833e-02f;
                } else {
                    return -3.76780448e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_54(const float *x) {
    if (x[3] < -3.62434611e-02f) {  /* VAR */
        if (x[5] < -4.81943178e+00f) {  /* LOG */
            if (x[2] < -9.95433569e-01f) {  /* WAMP */
                if (x[4] < -4.94039655e-01f) {  /* SNR */
                    return 4.50341031e-02f;
                } else {
                    return -3.68114673e-02f;
                }
            } else {
                if (x[1] < -3.68447229e-02f) {  /* WL */
                    return 1.14229433e-01f;
                } else {
                    return 1.12162903e-03f;
                }
            }
        } else {
            if (x[1] < -3.50219570e-02f) {  /* WL */
                if (x[2] < 1.91231871e+00f) {  /* WAMP */
                    return -1.04670413e-03f;
                } else {
                    return -7.35605508e-02f;
                }
            } else {
                if (x[2] < 1.69890559e+00f) {  /* WAMP */
                    return -5.86734228e-02f;
                } else {
                    return 6.21641949e-02f;
                }
            }
        }
    } else {
        if (x[1] < -3.40786092e-02f) {  /* WL */
            if (x[14] < -5.23453020e-03f) {  /* MeanFreq */
                if (x[2] < 2.09905505e+00f) {  /* WAMP */
                    return -1.72018960e-01f;
                } else {
                    return -3.79234888e-02f;
                }
            } else {
                if (x[1] < -3.62543650e-02f) {  /* WL */
                    return 4.47163545e-02f;
                } else {
                    return -5.83282597e-02f;
                }
            }
        } else {
            if (x[9] < -3.61460298e-02f) {  /* E_Low */
                if (x[2] < 2.44585133e+00f) {  /* WAMP */
                    return 9.18022022e-02f;
                } else {
                    return 2.61114538e-02f;
                }
            } else {
                if (x[13] < -6.29870594e-01f) {  /* SpectralKurtosis */
                    return -1.39645431e-02f;
                } else {
                    return -1.29573166e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_55(const float *x) {
    if (x[1] < -3.68358828e-02f) {  /* WL */
        if (x[13] < 3.64703685e-01f) {  /* SpectralKurtosis */
            if (x[14] < -5.03562167e-02f) {  /* MeanFreq */
                if (x[10] < -5.06316200e-02f) {  /* R_Mid */
                    return -8.00261945e-02f;
                } else {
                    return -1.77851189e-02f;
                }
            } else {
                if (x[14] < -5.03522158e-02f) {  /* MeanFreq */
                    return 9.86576825e-02f;
                } else {
                    return -6.00026362e-03f;
                }
            }
        } else {
            if (x[1] < -3.73004153e-02f) {  /* WL */
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 2.03142241e-02f;
                } else {
                    return -6.89708535e-03f;
                }
            } else {
                if (x[7] < -3.51393186e-02f) {  /* TFR_MeanEnergy */
                    return 1.94767825e-02f;
                } else {
                    return 1.25775993e-01f;
                }
            }
        }
    } else {
        if (x[13] < -5.29653728e-01f) {  /* SpectralKurtosis */
            if (x[1] < -3.67828310e-02f) {  /* WL */
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return -1.27053754e-02f;
                } else {
                    return 6.93508014e-02f;
                }
            } else {
                if (x[13] < -5.68953574e-01f) {  /* SpectralKurtosis */
                    return 4.25639190e-03f;
                } else {
                    return -1.62983034e-02f;
                }
            }
        } else {
            if (x[14] < -4.88331728e-02f) {  /* MeanFreq */
                if (x[5] < 4.63547111e-01f) {  /* LOG */
                    return 2.94280797e-02f;
                } else {
                    return -4.11686227e-02f;
                }
            } else {
                if (x[5] < 7.81269193e-01f) {  /* LOG */
                    return 1.22200966e-01f;
                } else {
                    return -5.06357364e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_56(const float *x) {
    if (x[1] < -3.50926928e-02f) {  /* WL */
        if (x[12] < 4.81761731e-02f) {  /* SpectralEntropy */
            if (x[13] < -6.09025300e-01f) {  /* SpectralKurtosis */
                if (x[10] < -4.97770198e-02f) {  /* R_Mid */
                    return -4.13576886e-02f;
                } else {
                    return 3.60138752e-02f;
                }
            } else {
                if (x[5] < 4.83148384e+00f) {  /* LOG */
                    return -3.59288213e-04f;
                } else {
                    return -9.17699486e-02f;
                }
            }
        } else {
            if (x[14] < -1.69531535e-02f) {  /* MeanFreq */
                if (x[1] < -3.62047143e-02f) {  /* WL */
                    return -8.72090980e-02f;
                } else {
                    return -2.71074902e-02f;
                }
            } else {
                if (x[1] < -3.66250388e-02f) {  /* WL */
                    return 9.15682837e-02f;
                } else {
                    return 1.26490919e-02f;
                }
            }
        }
    } else {
        if (x[6] < -3.62375416e-02f) {  /* SSI */
            if (x[12] < 3.91051061e-02f) {  /* SpectralEntropy */
                return -4.24671024e-02f;
            } else {
                if (x[10] < -1.40441740e-02f) {  /* R_Mid */
                    return 6.14896193e-02f;
                } else {
                    return -1.35232108e-02f;
                }
            }
        } else {
            if (x[0] < -4.94142342e-03f) {  /* MAV */
                if (x[14] < -3.27843390e-02f) {  /* MeanFreq */
                    return -2.95941774e-02f;
                } else {
                    return -1.26421973e-01f;
                }
            } else {
                if (x[13] < -6.24920428e-01f) {  /* SpectralKurtosis */
                    return 2.55661495e-02f;
                } else {
                    return -5.40258363e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_57(const float *x) {
    if (x[2] < 2.12573171e+00f) {  /* WAMP */
        if (x[0] < 4.67124060e-02f) {  /* MAV */
            if (x[7] < -3.51398326e-02f) {  /* TFR_MeanEnergy */
                if (x[4] < 2.18982935e+00f) {  /* SNR */
                    return -6.35442231e-03f;
                } else {
                    return 7.40688369e-02f;
                }
            } else {
                if (x[9] < -3.62521969e-02f) {  /* E_Low */
                    return 9.57504287e-03f;
                } else {
                    return -4.62868763e-03f;
                }
            }
        } else {
            if (x[5] < 7.59880185e-01f) {  /* LOG */
                if (x[2] < -5.68607569e-01f) {  /* WAMP */
                    return -6.40377402e-02f;
                } else {
                    return 8.71803090e-02f;
                }
            } else {
                if (x[3] < -3.62440981e-02f) {  /* VAR */
                    return -8.00598040e-03f;
                } else {
                    return -7.81824216e-02f;
                }
            }
        }
    } else {
        if (x[7] < -3.51306610e-02f) {  /* TFR_MeanEnergy */
            if (x[3] < -3.62434052e-02f) {  /* VAR */
                if (x[13] < -6.28083348e-01f) {  /* SpectralKurtosis */
                    return -8.69046003e-02f;
                } else {
                    return 6.25039488e-02f;
                }
            } else {
                return 9.08330902e-02f;
            }
        } else {
            if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                if (x[1] < -3.38970125e-02f) {  /* WL */
                    return -1.27959311e-01f;
                } else {
                    return 1.95280518e-02f;
                }
            } else {
                if (x[2] < 2.33914471e+00f) {  /* WAMP */
                    return 8.47129077e-02f;
                } else {
                    return 1.52522763e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_58(const float *x) {
    if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
        if (x[7] < -3.51415239e-02f) {  /* TFR_MeanEnergy */
            if (x[10] < -5.07096276e-02f) {  /* R_Mid */
                if (x[14] < -5.04163578e-02f) {  /* MeanFreq */
                    return -7.24507216e-03f;
                } else {
                    return 1.13923781e-01f;
                }
            } else {
                if (x[2] < -7.82020569e-01f) {  /* WAMP */
                    return -4.05348241e-02f;
                } else {
                    return -1.14565960e-03f;
                }
            }
        } else {
            if (x[7] < -3.51390168e-02f) {  /* TFR_MeanEnergy */
                if (x[4] < 2.18982935e+00f) {  /* SNR */
                    return 1.19393291e-02f;
                } else {
                    return 8.10602233e-02f;
                }
            } else {
                if (x[12] < -9.58702788e-02f) {  /* SpectralEntropy */
                    return -3.59332189e-02f;
                } else {
                    return 4.07898054e-03f;
                }
            }
        }
    } else {
        if (x[10] < -5.07192351e-02f) {  /* R_Mid */
            if (x[14] < -5.04486859e-02f) {  /* MeanFreq */
                if (x[4] < 1.62340498e+00f) {  /* SNR */
                    return 2.35473718e-02f;
                } else {
                    return -2.80943569e-02f;
                }
            } else {
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return -4.72571254e-02f;
                } else {
                    return 1.02057327e-02f;
                }
            }
        } else {
            if (x[0] < -1.95803400e-02f) {  /* MAV */
                if (x[9] < -3.61934863e-02f) {  /* E_Low */
                    return -8.96133773e-04f;
                } else {
                    return 6.26834854e-02f;
                }
            } else {
                if (x[6] < -3.62391062e-02f) {  /* SSI */
                    return -3.83201167e-02f;
                } else {
                    return 5.77333849e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_59(const float *x) {
    if (x[12] < -9.53515694e-02f) {  /* SpectralEntropy */
        if (x[2] < -7.28667319e-01f) {  /* WAMP */
            if (x[13] < 4.17913109e-01f) {  /* SpectralKurtosis */
                if (x[10] < -5.04002497e-02f) {  /* R_Mid */
                    return -4.38317582e-02f;
                } else {
                    return 4.65434976e-02f;
                }
            } else {
                if (x[1] < -3.72854508e-02f) {  /* WL */
                    return -4.14323040e-05f;
                } else {
                    return 1.25747070e-01f;
                }
            }
        } else {
            if (x[5] < -3.20150197e-01f) {  /* LOG */
                if (x[10] < -5.06015830e-02f) {  /* R_Mid */
                    return -6.22258857e-02f;
                } else {
                    return 1.46771818e-01f;
                }
            } else {
                if (x[1] < -3.73616256e-02f) {  /* WL */
                    return 2.71339510e-02f;
                } else {
                    return -7.92425573e-02f;
                }
            }
        }
    } else {
        if (x[14] < -5.03522158e-02f) {  /* MeanFreq */
            if (x[1] < -3.73242199e-02f) {  /* WL */
                if (x[14] < -5.03704995e-02f) {  /* MeanFreq */
                    return 4.03881408e-02f;
                } else {
                    return -3.40378471e-02f;
                }
            } else {
                if (x[13] < 4.53442365e-01f) {  /* SpectralKurtosis */
                    return 1.31294698e-01f;
                } else {
                    return 4.16712761e-02f;
                }
            }
        } else {
            if (x[2] < -8.88727069e-01f) {  /* WAMP */
                if (x[5] < -3.93775105e-02f) {  /* LOG */
                    return -7.62411579e-03f;
                } else {
                    return -4.94493172e-02f;
                }
            } else {
                if (x[12] < -9.44075808e-02f) {  /* SpectralEntropy */
                    return -3.50829028e-02f;
                } else {
                    return 4.31325473e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_60(const float *x) {
    if (x[9] < -3.62521969e-02f) {  /* E_Low */
        if (x[6] < -3.62451226e-02f) {  /* SSI */
            if (x[10] < -5.08396998e-02f) {  /* R_Mid */
                return 1.48312762e-01f;
            } else {
                if (x[10] < -5.07192351e-02f) {  /* R_Mid */
                    return -1.72193050e-02f;
                } else {
                    return 1.90632732e-03f;
                }
            }
        } else {
            if (x[14] < -5.05093522e-02f) {  /* MeanFreq */
                if (x[7] < -3.51392701e-02f) {  /* TFR_MeanEnergy */
                    return 2.51995865e-03f;
                } else {
                    return -1.01522855e-01f;
                }
            } else {
                if (x[8] < -3.62461023e-02f) {  /* E_Total */
                    return 6.48886189e-02f;
                } else {
                    return 1.36572812e-02f;
                }
            }
        }
    } else {
        if (x[4] < 1.52368701e+00f) {  /* SNR */
            if (x[6] < -3.62444557e-02f) {  /* SSI */
                if (x[14] < -4.90774028e-02f) {  /* MeanFreq */
                    return -1.71261858e-02f;
                } else {
                    return -1.28768966e-01f;
                }
            } else {
                if (x[7] < -3.51383947e-02f) {  /* TFR_MeanEnergy */
                    return 2.70830188e-02f;
                } else {
                    return -3.25088832e-03f;
                }
            }
        } else {
            if (x[2] < -9.68756974e-01f) {  /* WAMP */
                if (x[10] < -5.08213229e-02f) {  /* R_Mid */
                    return 4.35593538e-02f;
                } else {
                    return -6.77505806e-02f;
                }
            } else {
                return -1.02633670e-01f;
            }
        }
    }
}

static inline float xgb_tree_61(const float *x) {
    if (x[4] < 1.82875657e+00f) {  /* SNR */
        if (x[14] < -5.04824519e-02f) {  /* MeanFreq */
            if (x[4] < 1.35618246e+00f) {  /* SNR */
                if (x[3] < -3.62441279e-02f) {  /* VAR */
                    return -9.95963141e-02f;
                } else {
                    return -2.39964090e-02f;
                }
            } else {
                if (x[7] < -3.51392031e-02f) {  /* TFR_MeanEnergy */
                    return 5.77391796e-02f;
                } else {
                    return 1.04502365e-02f;
                }
            }
        } else {
            if (x[13] < -4.63768780e-01f) {  /* SpectralKurtosis */
                if (x[14] < -4.96292524e-02f) {  /* MeanFreq */
                    return 4.12087478e-02f;
                } else {
                    return 1.06921466e-03f;
                }
            } else {
                if (x[12] < -6.50291964e-02f) {  /* SpectralEntropy */
                    return -4.61484538e-03f;
                } else {
                    return 1.12204231e-01f;
                }
            }
        }
    } else {
        if (x[0] < -3.95539626e-02f) {  /* MAV */
            if (x[13] < 1.16086781e+00f) {  /* SpectralKurtosis */
                if (x[4] < 2.18982935e+00f) {  /* SNR */
                    return -9.45604518e-02f;
                } else {
                    return -8.60782620e-03f;
                }
            } else {
                if (x[1] < -3.74037959e-02f) {  /* WL */
                    return 1.39177507e-02f;
                } else {
                    return 1.65945694e-01f;
                }
            }
        } else {
            if (x[14] < -5.05093522e-02f) {  /* MeanFreq */
                if (x[0] < -3.80068049e-02f) {  /* MAV */
                    return -1.09721385e-01f;
                } else {
                    return -4.36212197e-02f;
                }
            } else {
                if (x[13] < 2.21051526e+00f) {  /* SpectralKurtosis */
                    return -8.52490738e-02f;
                } else {
                    return 1.04511287e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_62(const float *x) {
    if (x[10] < -5.07192351e-02f) {  /* R_Mid */
        if (x[13] < 4.34184343e-01f) {  /* SpectralKurtosis */
            if (x[0] < -4.54809703e-02f) {  /* MAV */
                if (x[10] < -5.07556573e-02f) {  /* R_Mid */
                    return 1.00879453e-01f;
                } else {
                    return -8.81516282e-03f;
                }
            } else {
                if (x[0] < -1.61885489e-02f) {  /* MAV */
                    return -5.55448644e-02f;
                } else {
                    return 1.30133465e-01f;
                }
            }
        } else {
            if (x[1] < -3.72888483e-02f) {  /* WL */
                if (x[14] < -5.03961258e-02f) {  /* MeanFreq */
                    return -1.86369449e-04f;
                } else {
                    return -3.44660543e-02f;
                }
            } else {
                if (x[0] < -3.08925323e-02f) {  /* MAV */
                    return 1.13028347e-01f;
                } else {
                    return -6.01231232e-02f;
                }
            }
        }
    } else {
        if (x[1] < -3.74554843e-02f) {  /* WL */
            if (x[10] < -4.85422835e-02f) {  /* R_Mid */
                if (x[12] < -9.78989899e-02f) {  /* SpectralEntropy */
                    return -6.62392238e-03f;
                } else {
                    return 1.30526975e-01f;
                }
            } else {
                if (x[13] < -5.51105559e-01f) {  /* SpectralKurtosis */
                    return 1.64190251e-02f;
                } else {
                    return -7.50226900e-02f;
                }
            }
        } else {
            if (x[7] < -3.51410992e-02f) {  /* TFR_MeanEnergy */
                if (x[9] < -3.62738967e-02f) {  /* E_Low */
                    return -6.90199900e-03f;
                } else {
                    return -2.14930817e-01f;
                }
            } else {
                if (x[7] < -3.51409428e-02f) {  /* TFR_MeanEnergy */
                    return 4.61824983e-02f;
                } else {
                    return 2.51900521e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_63(const float *x) {
    if (x[6] < -3.62408496e-02f) {  /* SSI */
        if (x[4] < -5.68556666e-01f) {  /* SNR */
            if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                if (x[2] < 1.53884590e+00f) {  /* WAMP */
                    return -2.04084124e-02f;
                } else {
                    return -2.07187817e-01f;
                }
            } else {
                if (x[14] < -4.59191911e-02f) {  /* MeanFreq */
                    return 2.56008599e-02f;
                } else {
                    return 3.29248462e-04f;
                }
            }
        } else {
            if (x[2] < 8.45253587e-01f) {  /* WAMP */
                if (x[5] < -1.74581444e+00f) {  /* LOG */
                    return 3.47615518e-02f;
                } else {
                    return 9.11197567e-04f;
                }
            } else {
                if (x[6] < -3.62412147e-02f) {  /* SSI */
                    return -7.78705329e-02f;
                } else {
                    return 3.98232006e-02f;
                }
            }
        }
    } else {
        if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
            if (x[5] < 6.48958266e-01f) {  /* LOG */
                if (x[14] < -4.33010422e-02f) {  /* MeanFreq */
                    return -1.30997270e-01f;
                } else {
                    return -3.70810665e-02f;
                }
            } else {
                if (x[6] < -3.62391062e-02f) {  /* SSI */
                    return 7.14472830e-02f;
                } else {
                    return 6.11568941e-03f;
                }
            }
        } else {
            if (x[4] < -3.27857137e-01f) {  /* SNR */
                if (x[2] < 1.51216924e+00f) {  /* WAMP */
                    return 6.02856651e-03f;
                } else {
                    return -3.25085334e-02f;
                }
            } else {
                if (x[5] < 1.04134607e+00f) {  /* LOG */
                    return -7.67902136e-02f;
                } else {
                    return 7.93095678e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_64(const float *x) {
    if (x[12] < -9.33036879e-02f) {  /* SpectralEntropy */
        if (x[2] < -7.28667319e-01f) {  /* WAMP */
            if (x[0] < -9.27947909e-02f) {  /* MAV */
                if (x[10] < -5.05131334e-02f) {  /* R_Mid */
                    return 1.45836264e-01f;
                } else {
                    return 4.00527418e-02f;
                }
            } else {
                if (x[4] < 6.05665863e-01f) {  /* SNR */
                    return -2.88305320e-02f;
                } else {
                    return 1.32322474e-03f;
                }
            }
        } else {
            if (x[0] < -5.08316159e-02f) {  /* MAV */
                return 1.31721899e-01f;
            } else {
                if (x[7] < -3.51384915e-02f) {  /* TFR_MeanEnergy */
                    return -6.87011555e-02f;
                } else {
                    return 5.79426158e-03f;
                }
            }
        }
    } else {
        if (x[14] < -5.03199585e-02f) {  /* MeanFreq */
            if (x[13] < 6.97020054e-01f) {  /* SpectralKurtosis */
                if (x[4] < 5.12316346e-01f) {  /* SNR */
                    return 1.04922034e-01f;
                } else {
                    return -1.58449505e-02f;
                }
            } else {
                if (x[4] < 4.33944255e-01f) {  /* SNR */
                    return -8.37252960e-02f;
                } else {
                    return 4.51494418e-02f;
                }
            }
        } else {
            if (x[2] < -9.68756974e-01f) {  /* WAMP */
                if (x[0] < -1.04478806e-01f) {  /* MAV */
                    return 2.18398012e-02f;
                } else {
                    return -8.59067962e-02f;
                }
            } else {
                if (x[2] < 1.43213940e+00f) {  /* WAMP */
                    return 5.24526974e-03f;
                } else {
                    return -1.05344467e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_65(const float *x) {
    if (x[5] < -4.81943178e+00f) {  /* LOG */
        if (x[3] < -3.62441167e-02f) {  /* VAR */
            return 1.19159877e-01f;
        } else {
            if (x[1] < -3.74554843e-02f) {  /* WL */
                if (x[4] < -4.94039655e-01f) {  /* SNR */
                    return 4.31135893e-02f;
                } else {
                    return -5.57305887e-02f;
                }
            } else {
                if (x[4] < -9.96244848e-01f) {  /* SNR */
                    return -8.61002132e-03f;
                } else {
                    return 8.89242589e-02f;
                }
            }
        }
    } else {
        if (x[0] < -4.54809703e-02f) {  /* MAV */
            if (x[9] < -3.62785198e-02f) {  /* E_Low */
                if (x[2] < -9.68756974e-01f) {  /* WAMP */
                    return -4.77944575e-02f;
                } else {
                    return -3.50541482e-03f;
                }
            } else {
                if (x[7] < -3.51413675e-02f) {  /* TFR_MeanEnergy */
                    return 7.02315792e-02f;
                } else {
                    return -1.04507722e-01f;
                }
            }
        } else {
            if (x[5] < -2.03995481e-02f) {  /* LOG */
                if (x[13] < -4.96839792e-01f) {  /* SpectralKurtosis */
                    return 2.10888311e-02f;
                } else {
                    return 1.78791396e-03f;
                }
            } else {
                if (x[13] < -5.88590384e-01f) {  /* SpectralKurtosis */
                    return 1.02474019e-02f;
                } else {
                    return -8.04564171e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_66(const float *x) {
    if (x[4] < 1.82875657e+00f) {  /* SNR */
        if (x[10] < -5.08396998e-02f) {  /* R_Mid */
            if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                return 2.30418723e-02f;
            } else {
                return 1.59230709e-01f;
            }
        } else {
            if (x[1] < -3.72249186e-02f) {  /* WL */
                if (x[13] < 3.64703685e-01f) {  /* SpectralKurtosis */
                    return -1.77567601e-02f;
                } else {
                    return 4.29732213e-03f;
                }
            } else {
                if (x[4] < 5.55610001e-01f) {  /* SNR */
                    return 4.07121936e-03f;
                } else {
                    return -3.14929187e-02f;
                }
            }
        }
    } else {
        if (x[7] < -3.51394638e-02f) {  /* TFR_MeanEnergy */
            if (x[4] < 2.18982935e+00f) {  /* SNR */
                if (x[7] < -3.51394787e-02f) {  /* TFR_MeanEnergy */
                    return -6.58132136e-02f;
                } else {
                    return 1.09540462e-01f;
                }
            } else {
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return 9.46023092e-02f;
                } else {
                    return -6.30220920e-02f;
                }
            }
        } else {
            if (x[1] < -3.74309979e-02f) {  /* WL */
                if (x[0] < -3.87929790e-02f) {  /* MAV */
                    return -1.07827522e-01f;
                } else {
                    return -5.00838049e-02f;
                }
            } else {
                if (x[0] < -3.84410731e-02f) {  /* MAV */
                    return 8.07436332e-02f;
                } else {
                    return -8.99279937e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_67(const float *x) {
    if (x[6] < -3.62451226e-02f) {  /* SSI */
        if (x[13] < 3.00952411e+00f) {  /* SpectralKurtosis */
            if (x[4] < 2.37086248e+00f) {  /* SNR */
                if (x[14] < -5.04960157e-02f) {  /* MeanFreq */
                    return -9.65786129e-02f;
                } else {
                    return -3.81148909e-03f;
                }
            } else {
                if (x[14] < -5.04990146e-02f) {  /* MeanFreq */
                    return 1.42834499e-01f;
                } else {
                    return 2.87962742e-02f;
                }
            }
        } else {
            if (x[7] < -3.51398140e-02f) {  /* TFR_MeanEnergy */
                if (x[14] < -5.05066253e-02f) {  /* MeanFreq */
                    return -8.10663700e-02f;
                } else {
                    return 6.37177676e-02f;
                }
            } else {
                if (x[2] < -9.95433569e-01f) {  /* WAMP */
                    return 1.47542730e-01f;
                } else {
                    return 4.62436043e-02f;
                }
            }
        }
    } else {
        if (x[4] < 1.82875657e+00f) {  /* SNR */
            if (x[7] < -3.51383947e-02f) {  /* TFR_MeanEnergy */
                if (x[4] < 6.62612736e-01f) {  /* SNR */
                    return 1.96948275e-02f;
                } else {
                    return 3.91575368e-03f;
                }
            } else {
                if (x[10] < -5.07706180e-02f) {  /* R_Mid */
                    return 4.33781147e-02f;
                } else {
                    return -2.60662218e-03f;
                }
            }
        } else {
            if (x[7] < -3.51394638e-02f) {  /* TFR_MeanEnergy */
                if (x[7] < -3.51394787e-02f) {  /* TFR_MeanEnergy */
                    return -2.85068434e-02f;
                } else {
                    return 1.36960134e-01f;
                }
            } else {
                if (x[4] < 2.52899075e+00f) {  /* SNR */
                    return -3.94709706e-02f;
                } else {
                    return -9.81862172e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_68(const float *x) {
    if (x[10] < -5.08336388e-02f) {  /* R_Mid */
        if (x[12] < -9.90521535e-02f) {  /* SpectralEntropy */
            if (x[5] < -7.56144896e-02f) {  /* LOG */
                if (x[4] < 2.18982935e+00f) {  /* SNR */
                    return -2.91005950e-02f;
                } else {
                    return 7.77353719e-02f;
                }
            } else {
                if (x[10] < -5.08396998e-02f) {  /* R_Mid */
                    return -8.96085799e-02f;
                } else {
                    return -2.75500701e-03f;
                }
            }
        } else {
            if (x[5] < -1.74312312e-02f) {  /* LOG */
                if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                    return 7.89029673e-02f;
                } else {
                    return 1.53180629e-01f;
                }
            } else {
                if (x[10] < -5.08363843e-02f) {  /* R_Mid */
                    return 6.99989498e-02f;
                } else {
                    return -7.18702450e-02f;
                }
            }
        }
    } else {
        if (x[14] < -4.54153642e-02f) {  /* MeanFreq */
            if (x[10] < -4.73805293e-02f) {  /* R_Mid */
                if (x[9] < -3.61635648e-02f) {  /* E_Low */
                    return -1.60775427e-03f;
                } else {
                    return 3.14108990e-02f;
                }
            } else {
                if (x[2] < 1.40546274e+00f) {  /* WAMP */
                    return 2.65870690e-02f;
                } else {
                    return -2.35944130e-02f;
                }
            }
        } else {
            if (x[1] < -3.60054336e-02f) {  /* WL */
                if (x[12] < 5.29933393e-01f) {  /* SpectralEntropy */
                    return -7.14573711e-02f;
                } else {
                    return 7.21632391e-02f;
                }
            } else {
                if (x[10] < -4.50112633e-02f) {  /* R_Mid */
                    return -2.14432497e-02f;
                } else {
                    return 1.15254661e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_69(const float *x) {
    if (x[1] < -3.53484228e-02f) {  /* WL */
        if (x[2] < 1.88564217e+00f) {  /* WAMP */
            if (x[4] < -1.08698773e+00f) {  /* SNR */
                if (x[6] < -3.62469070e-02f) {  /* SSI */
                    return 1.81416105e-02f;
                } else {
                    return -4.09757867e-02f;
                }
            } else {
                if (x[4] < -4.08575267e-01f) {  /* SNR */
                    return 1.06810583e-02f;
                } else {
                    return -1.34911330e-03f;
                }
            }
        } else {
            if (x[13] < -6.13503516e-01f) {  /* SpectralKurtosis */
                if (x[1] < -3.54348011e-02f) {  /* WL */
                    return -4.52979608e-03f;
                } else {
                    return 6.53747842e-02f;
                }
            } else {
                if (x[1] < -3.55293415e-02f) {  /* WL */
                    return -2.03901768e-01f;
                } else {
                    return -7.35246167e-02f;
                }
            }
        }
    } else {
        if (x[4] < -1.65028071e+00f) {  /* SNR */
            if (x[13] < -6.24920428e-01f) {  /* SpectralKurtosis */
                if (x[2] < 1.99234867e+00f) {  /* WAMP */
                    return -4.33807150e-02f;
                } else {
                    return 5.91667779e-02f;
                }
            } else {
                if (x[7] < -3.51378210e-02f) {  /* TFR_MeanEnergy */
                    return 3.44624370e-02f;
                } else {
                    return -1.90119207e-01f;
                }
            }
        } else {
            if (x[5] < -8.31438527e-02f) {  /* LOG */
                if (x[5] < -2.95457035e-01f) {  /* LOG */
                    return 6.06204048e-02f;
                } else {
                    return -7.10631534e-02f;
                }
            } else {
                if (x[10] < -4.93563786e-02f) {  /* R_Mid */
                    return -9.50173438e-02f;
                } else {
                    return 7.15230554e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_70(const float *x) {
    if (x[4] < 7.96598792e-01f) {  /* SNR */
        if (x[14] < -4.99593318e-02f) {  /* MeanFreq */
            if (x[2] < -8.88727069e-01f) {  /* WAMP */
                if (x[14] < -5.01481742e-02f) {  /* MeanFreq */
                    return -7.07868626e-03f;
                } else {
                    return -7.38403946e-02f;
                }
            } else {
                if (x[10] < -4.93563786e-02f) {  /* R_Mid */
                    return 1.45198051e-02f;
                } else {
                    return 1.12507157e-01f;
                }
            }
        } else {
            if (x[2] < -7.01990664e-01f) {  /* WAMP */
                if (x[4] < -1.35352716e-01f) {  /* SNR */
                    return -5.75579656e-03f;
                } else {
                    return -7.99703002e-02f;
                }
            } else {
                if (x[12] < -8.71591642e-02f) {  /* SpectralEntropy */
                    return -5.21108769e-02f;
                } else {
                    return 3.67807806e-04f;
                }
            }
        }
    } else {
        if (x[2] < -7.55343974e-01f) {  /* WAMP */
            if (x[4] < 8.08718443e-01f) {  /* SNR */
                if (x[10] < -5.07961437e-02f) {  /* R_Mid */
                    return -3.18783685e-03f;
                } else {
                    return -8.76607299e-02f;
                }
            } else {
                if (x[13] < -4.79845107e-02f) {  /* SpectralKurtosis */
                    return 7.78080747e-02f;
                } else {
                    return -1.75582047e-03f;
                }
            }
        } else {
            if (x[0] < -5.21450303e-02f) {  /* MAV */
                return 1.03077009e-01f;
            } else {
                if (x[4] < 8.20931137e-01f) {  /* SNR */
                    return 3.07635851e-02f;
                } else {
                    return -6.79158941e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_71(const float *x) {
    if (x[14] < -4.91066240e-02f) {  /* MeanFreq */
        if (x[14] < -4.94047999e-02f) {  /* MeanFreq */
            if (x[1] < -3.69236171e-02f) {  /* WL */
                if (x[4] < 3.31234068e-01f) {  /* SNR */
                    return 9.40277055e-03f;
                } else {
                    return -2.31395941e-03f;
                }
            } else {
                if (x[10] < -4.90278602e-02f) {  /* R_Mid */
                    return -2.68319678e-02f;
                } else {
                    return 3.92111652e-02f;
                }
            }
        } else {
            if (x[2] < 6.58517241e-01f) {  /* WAMP */
                if (x[6] < -3.62456404e-02f) {  /* SSI */
                    return -5.96918305e-03f;
                } else {
                    return 4.95041981e-02f;
                }
            } else {
                if (x[13] < -5.52980542e-01f) {  /* SpectralKurtosis */
                    return -1.73406154e-02f;
                } else {
                    return -1.14839606e-01f;
                }
            }
        }
    } else {
        if (x[1] < -3.62373628e-02f) {  /* WL */
            if (x[2] < 1.08534324e+00f) {  /* WAMP */
                if (x[12] < -7.08012804e-02f) {  /* SpectralEntropy */
                    return -6.49781674e-02f;
                } else {
                    return -7.95684662e-03f;
                }
            } else {
                if (x[6] < -3.62391062e-02f) {  /* SSI */
                    return -6.62092194e-02f;
                } else {
                    return 6.45228028e-02f;
                }
            }
        } else {
            if (x[2] < 1.43213940e+00f) {  /* WAMP */
                if (x[4] < -9.96244848e-01f) {  /* SNR */
                    return -1.12731371e-03f;
                } else {
                    return 4.89402041e-02f;
                }
            } else {
                if (x[3] < -3.62440050e-02f) {  /* VAR */
                    return -6.63454309e-02f;
                } else {
                    return 5.12939459e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_72(const float *x) {
    if (x[7] < -3.51386331e-02f) {  /* TFR_MeanEnergy */
        if (x[7] < -3.51398736e-02f) {  /* TFR_MeanEnergy */
            if (x[2] < 8.71930242e-01f) {  /* WAMP */
                if (x[9] < -3.62624042e-02f) {  /* E_Low */
                    return -5.40115777e-03f;
                } else {
                    return -1.25466719e-01f;
                }
            } else {
                if (x[3] < -3.62439491e-02f) {  /* VAR */
                    return 4.67822589e-02f;
                } else {
                    return -3.99905769e-03f;
                }
            }
        } else {
            if (x[6] < -3.62452529e-02f) {  /* SSI */
                if (x[1] < -3.69317792e-02f) {  /* WL */
                    return 5.73561750e-02f;
                } else {
                    return -1.67347807e-02f;
                }
            } else {
                if (x[9] < -3.62647735e-02f) {  /* E_Low */
                    return -1.55774459e-01f;
                } else {
                    return 3.40517913e-03f;
                }
            }
        }
    } else {
        if (x[7] < -3.51385251e-02f) {  /* TFR_MeanEnergy */
            if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                if (x[1] < -3.69725898e-02f) {  /* WL */
                    return -4.38934229e-02f;
                } else {
                    return 4.47311997e-02f;
                }
            } else {
                if (x[12] < -8.71591642e-02f) {  /* SpectralEntropy */
                    return -2.63455510e-02f;
                } else {
                    return -1.86005741e-01f;
                }
            }
        } else {
            if (x[7] < -3.51384319e-02f) {  /* TFR_MeanEnergy */
                if (x[1] < -3.72759290e-02f) {  /* WL */
                    return -2.22483650e-02f;
                } else {
                    return 6.55635372e-02f;
                }
            } else {
                if (x[2] < -7.55343974e-01f) {  /* WAMP */
                    return -2.58758366e-02f;
                } else {
                    return -2.95059034e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_73(const float *x) {
    if (x[12] < 6.24457002e-02f) {  /* SpectralEntropy */
        if (x[1] < -3.59360613e-02f) {  /* WL */
            if (x[14] < -4.54153642e-02f) {  /* MeanFreq */
                if (x[13] < -5.95062017e-01f) {  /* SpectralKurtosis */
                    return 2.11026128e-02f;
                } else {
                    return -7.68888276e-04f;
                }
            } else {
                if (x[6] < -3.62454168e-02f) {  /* SSI */
                    return -1.78671088e-02f;
                } else {
                    return -1.11875668e-01f;
                }
            }
        } else {
            if (x[3] < -3.62440050e-02f) {  /* VAR */
                if (x[0] < -2.12749261e-02f) {  /* MAV */
                    return -7.90916234e-02f;
                } else {
                    return 1.39683560e-02f;
                }
            } else {
                if (x[13] < -6.28739893e-01f) {  /* SpectralKurtosis */
                    return -9.37241539e-02f;
                } else {
                    return 3.25738266e-02f;
                }
            }
        }
    } else {
        if (x[13] < -6.26930475e-01f) {  /* SpectralKurtosis */
            if (x[5] < 2.11366519e-01f) {  /* LOG */
                if (x[0] < -1.66312024e-01f) {  /* MAV */
                    return -5.73912822e-02f;
                } else {
                    return 7.35724345e-02f;
                }
            } else {
                if (x[12] < 1.45064801e-01f) {  /* SpectralEntropy */
                    return 6.74525052e-02f;
                } else {
                    return -4.23755758e-02f;
                }
            }
        } else {
            if (x[12] < 2.46237680e-01f) {  /* SpectralEntropy */
                if (x[1] < -3.50219570e-02f) {  /* WL */
                    return -4.25028615e-02f;
                } else {
                    return 1.93353202e-02f;
                }
            } else {
                if (x[1] < -3.40786092e-02f) {  /* WL */
                    return -1.85889259e-01f;
                } else {
                    return 2.11125035e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_74(const float *x) {
    if (x[2] < -9.95433569e-01f) {  /* WAMP */
        if (x[0] < -4.00798470e-02f) {  /* MAV */
            if (x[7] < -3.51408869e-02f) {  /* TFR_MeanEnergy */
                if (x[4] < 1.41474891e+00f) {  /* SNR */
                    return -2.82885432e-02f;
                } else {
                    return 6.54621944e-02f;
                }
            } else {
                if (x[10] < -5.03421463e-02f) {  /* R_Mid */
                    return 8.62819925e-02f;
                } else {
                    return -5.94047867e-02f;
                }
            }
        } else {
            if (x[4] < 2.37086248e+00f) {  /* SNR */
                if (x[10] < -5.07870540e-02f) {  /* R_Mid */
                    return 2.98346411e-02f;
                } else {
                    return -2.12468114e-02f;
                }
            } else {
                return -9.55014452e-02f;
            }
        }
    } else {
        if (x[13] < 2.45617104e+00f) {  /* SpectralKurtosis */
            if (x[2] < 2.04570198e+00f) {  /* WAMP */
                if (x[3] < -3.62434611e-02f) {  /* VAR */
                    return 3.71694041e-04f;
                } else {
                    return -4.39400189e-02f;
                }
            } else {
                if (x[0] < -2.42815707e-02f) {  /* MAV */
                    return 6.62163869e-02f;
                } else {
                    return -3.46487085e-03f;
                }
            }
        } else {
            if (x[7] < -3.51392031e-02f) {  /* TFR_MeanEnergy */
                if (x[10] < -5.08003086e-02f) {  /* R_Mid */
                    return -1.68527048e-02f;
                } else {
                    return 1.14265971e-01f;
                }
            } else {
                if (x[0] < -3.57637629e-02f) {  /* MAV */
                    return -1.00809850e-01f;
                } else {
                    return -1.47064764e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_75(const float *x) {
    if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
        if (x[1] < -3.73915508e-02f) {  /* WL */
            if (x[0] < -5.26270978e-02f) {  /* MAV */
                if (x[13] < 1.06494689e+00f) {  /* SpectralKurtosis */
                    return 6.42463490e-02f;
                } else {
                    return -2.24230625e-02f;
                }
            } else {
                if (x[2] < -9.95433569e-01f) {  /* WAMP */
                    return 1.21084349e-02f;
                } else {
                    return -2.94338372e-02f;
                }
            }
        } else {
            if (x[13] < 3.45593154e-01f) {  /* SpectralKurtosis */
                if (x[1] < -3.72793302e-02f) {  /* WL */
                    return -1.88845303e-02f;
                } else {
                    return 3.81816365e-03f;
                }
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 6.92871436e-02f;
                } else {
                    return 7.50991469e-03f;
                }
            }
        }
    } else {
        if (x[13] < -6.18349969e-01f) {  /* SpectralKurtosis */
            if (x[2] < 9.83080864e-02f) {  /* WAMP */
                if (x[2] < -7.55343974e-01f) {  /* WAMP */
                    return -3.65279317e-02f;
                } else {
                    return -1.54255927e-01f;
                }
            } else {
                if (x[10] < -4.95533496e-02f) {  /* R_Mid */
                    return -5.07536829e-02f;
                } else {
                    return 3.56109552e-02f;
                }
            }
        } else {
            if (x[2] < 1.53884590e+00f) {  /* WAMP */
                if (x[3] < -3.62439193e-02f) {  /* VAR */
                    return -2.87480210e-03f;
                } else {
                    return -5.51819019e-02f;
                }
            } else {
                if (x[3] < -3.62439752e-02f) {  /* VAR */
                    return -7.17192963e-02f;
                } else {
                    return -2.74324580e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_76(const float *x) {
    if (x[13] < -5.88590384e-01f) {  /* SpectralKurtosis */
        if (x[3] < -3.62439491e-02f) {  /* VAR */
            if (x[2] < 1.64555240e+00f) {  /* WAMP */
                if (x[1] < -3.61251384e-02f) {  /* WL */
                    return 1.06739411e-02f;
                } else {
                    return 6.51305914e-02f;
                }
            } else {
                if (x[8] < -3.62403095e-02f) {  /* E_Total */
                    return -8.26121569e-02f;
                } else {
                    return 1.71864163e-02f;
                }
            }
        } else {
            if (x[2] < 1.69890559e+00f) {  /* WAMP */
                if (x[4] < -9.65552390e-01f) {  /* SNR */
                    return -8.76804255e-03f;
                } else {
                    return -8.62127095e-02f;
                }
            } else {
                if (x[2] < 1.80561221e+00f) {  /* WAMP */
                    return 5.96480183e-02f;
                } else {
                    return 9.56840999e-03f;
                }
            }
        }
    } else {
        if (x[2] < 1.11201990e+00f) {  /* WAMP */
            if (x[1] < -3.63992304e-02f) {  /* WL */
                if (x[10] < -4.38137725e-02f) {  /* R_Mid */
                    return -6.83167833e-04f;
                } else {
                    return -1.26296923e-01f;
                }
            } else {
                if (x[10] < -4.93563786e-02f) {  /* R_Mid */
                    return -1.78504940e-02f;
                } else {
                    return 5.47228567e-02f;
                }
            }
        } else {
            if (x[6] < -3.62391062e-02f) {  /* SSI */
                if (x[0] < -5.31583168e-02f) {  /* MAV */
                    return 8.87168497e-02f;
                } else {
                    return -6.43419698e-02f;
                }
            } else {
                if (x[3] < -3.62440161e-02f) {  /* VAR */
                    return 1.37716815e-01f;
                } else {
                    return -9.53476802e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_77(const float *x) {
    if (x[7] < -3.51466835e-02f) {  /* TFR_MeanEnergy */
        if (x[8] < -3.62604596e-02f) {  /* E_Total */
            if (x[13] < -6.14188969e-01f) {  /* SpectralKurtosis */
                if (x[10] < -3.31971049e-02f) {  /* R_Mid */
                    return 9.41775516e-02f;
                } else {
                    return 2.36111227e-02f;
                }
            } else {
                if (x[13] < -6.04469478e-01f) {  /* SpectralKurtosis */
                    return -8.78105015e-02f;
                } else {
                    return 6.77879900e-03f;
                }
            }
        } else {
            if (x[10] < -4.94689420e-02f) {  /* R_Mid */
                if (x[12] < -7.98559561e-02f) {  /* SpectralEntropy */
                    return -4.91222814e-02f;
                } else {
                    return 5.21950722e-02f;
                }
            } else {
                if (x[6] < -3.62495929e-02f) {  /* SSI */
                    return -6.84241652e-02f;
                } else {
                    return 6.31369427e-02f;
                }
            }
        }
    } else {
        if (x[7] < -3.51437554e-02f) {  /* TFR_MeanEnergy */
            if (x[5] < -6.27905130e-01f) {  /* LOG */
                if (x[8] < -3.62517126e-02f) {  /* E_Total */
                    return 2.35045794e-02f;
                } else {
                    return -5.16487248e-02f;
                }
            } else {
                if (x[13] < 2.19011813e-01f) {  /* SpectralKurtosis */
                    return 1.27750620e-01f;
                } else {
                    return -7.20751062e-02f;
                }
            }
        } else {
            if (x[7] < -3.51429246e-02f) {  /* TFR_MeanEnergy */
                if (x[12] < 2.25239005e-02f) {  /* SpectralEntropy */
                    return -1.16113815e-02f;
                } else {
                    return -1.72772110e-01f;
                }
            } else {
                if (x[1] < -3.74010727e-02f) {  /* WL */
                    return -1.03238793e-02f;
                } else {
                    return 2.39251251e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_78(const float *x) {
    if (x[14] < -5.04824519e-02f) {  /* MeanFreq */
        if (x[6] < -3.62448730e-02f) {  /* SSI */
            if (x[1] < -3.74037959e-02f) {  /* WL */
                if (x[13] < 1.51893795e+00f) {  /* SpectralKurtosis */
                    return -5.38292713e-02f;
                } else {
                    return 3.95915173e-02f;
                }
            } else {
                if (x[13] < 1.03015113e+00f) {  /* SpectralKurtosis */
                    return -9.15908068e-03f;
                } else {
                    return 1.66666538e-01f;
                }
            }
        } else {
            if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                if (x[10] < -5.08119687e-02f) {  /* R_Mid */
                    return -2.95430608e-02f;
                } else {
                    return -1.03367545e-01f;
                }
            } else {
                if (x[14] < -5.05093522e-02f) {  /* MeanFreq */
                    return -9.20151696e-02f;
                } else {
                    return 2.15072706e-02f;
                }
            }
        }
    } else {
        if (x[11] < 5.80264390e-01f) {  /* TopFreq2 */
            if (x[1] < -3.73752266e-02f) {  /* WL */
                if (x[2] < -8.62050474e-01f) {  /* WAMP */
                    return -1.18876426e-02f;
                } else {
                    return -6.17671721e-02f;
                }
            } else {
                if (x[13] < 8.40160906e-01f) {  /* SpectralKurtosis */
                    return -2.13975227e-03f;
                } else {
                    return 2.74705496e-02f;
                }
            }
        } else {
            if (x[1] < -3.71732302e-02f) {  /* WL */
                if (x[2] < -9.42080319e-01f) {  /* WAMP */
                    return 7.08952993e-02f;
                } else {
                    return -4.43118662e-02f;
                }
            } else {
                if (x[0] < -5.31583168e-02f) {  /* MAV */
                    return -6.16222508e-02f;
                } else {
                    return 1.19694345e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_79(const float *x) {
    if (x[6] < -3.62452790e-02f) {  /* SSI */
        if (x[9] < -3.62645611e-02f) {  /* E_Low */
            if (x[12] < -9.92424935e-02f) {  /* SpectralEntropy */
                if (x[0] < -4.40174751e-02f) {  /* MAV */
                    return -6.04655445e-02f;
                } else {
                    return 1.02725916e-01f;
                }
            } else {
                if (x[5] < -4.81943178e+00f) {  /* LOG */
                    return 4.63834032e-02f;
                } else {
                    return -5.35738049e-03f;
                }
            }
        } else {
            if (x[1] < -3.71331014e-02f) {  /* WL */
                if (x[1] < -3.73038128e-02f) {  /* WL */
                    return -7.90443942e-02f;
                } else {
                    return 2.49239225e-02f;
                }
            } else {
                if (x[13] < -5.03881514e-01f) {  /* SpectralKurtosis */
                    return 2.88483631e-02f;
                } else {
                    return -1.44382328e-01f;
                }
            }
        }
    } else {
        if (x[8] < -3.62463742e-02f) {  /* E_Total */
            if (x[8] < -3.62465307e-02f) {  /* E_Total */
                if (x[13] < -6.21251106e-01f) {  /* SpectralKurtosis */
                    return 2.15509292e-02f;
                } else {
                    return -1.02444865e-01f;
                }
            } else {
                if (x[2] < -8.88727069e-01f) {  /* WAMP */
                    return 1.00025378e-01f;
                } else {
                    return 4.57299761e-02f;
                }
            }
        } else {
            if (x[0] < -1.95803400e-02f) {  /* MAV */
                if (x[6] < -3.62451226e-02f) {  /* SSI */
                    return -1.49919959e-02f;
                } else {
                    return 4.86613531e-03f;
                }
            } else {
                if (x[13] < -3.22974920e-01f) {  /* SpectralKurtosis */
                    return -1.84895545e-02f;
                } else {
                    return 3.57405245e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_80(const float *x) {
    if (x[14] < -5.04800528e-02f) {  /* MeanFreq */
        if (x[5] < -7.38709420e-02f) {  /* LOG */
            if (x[1] < -3.73888314e-02f) {  /* WL */
                if (x[13] < 1.51893795e+00f) {  /* SpectralKurtosis */
                    return -3.95499580e-02f;
                } else {
                    return 3.99246924e-02f;
                }
            } else {
                if (x[4] < 1.91054463e+00f) {  /* SNR */
                    return -7.94174429e-03f;
                } else {
                    return 1.36670455e-01f;
                }
            }
        } else {
            if (x[4] < 1.74811363e+00f) {  /* SNR */
                if (x[10] < -5.07870540e-02f) {  /* R_Mid */
                    return 3.71956266e-02f;
                } else {
                    return -4.70545776e-02f;
                }
            } else {
                if (x[12] < -9.94796827e-02f) {  /* SpectralEntropy */
                    return -9.50940177e-02f;
                } else {
                    return -3.41214575e-02f;
                }
            }
        }
    } else {
        if (x[12] < -9.86877456e-02f) {  /* SpectralEntropy */
            if (x[14] < -5.04774898e-02f) {  /* MeanFreq */
                return 1.16864806e-02f;
            } else {
                if (x[1] < -3.73344198e-02f) {  /* WL */
                    return -9.91301313e-02f;
                } else {
                    return -2.57345289e-02f;
                }
            }
        } else {
            if (x[13] < 1.23580539e+00f) {  /* SpectralKurtosis */
                if (x[1] < -3.74554843e-02f) {  /* WL */
                    return 6.93582818e-02f;
                } else {
                    return 1.54115784e-04f;
                }
            } else {
                if (x[4] < 1.06509686e+00f) {  /* SNR */
                    return -5.10791801e-02f;
                } else {
                    return -3.81677528e-04f;
                }
            }
        }
    }
}

static inline float xgb_tree_81(const float *x) {
    if (x[14] < -4.98700067e-02f) {  /* MeanFreq */
        if (x[2] < -3.01841319e-01f) {  /* WAMP */
            if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                if (x[1] < -3.73425819e-02f) {  /* WL */
                    return 3.35181202e-03f;
                } else {
                    return 3.68178897e-02f;
                }
            } else {
                if (x[10] < -5.06735072e-02f) {  /* R_Mid */
                    return -1.04801776e-02f;
                } else {
                    return 7.75369210e-03f;
                }
            }
        } else {
            if (x[10] < -5.00742309e-02f) {  /* R_Mid */
                if (x[0] < -4.21157293e-02f) {  /* MAV */
                    return -8.69831592e-02f;
                } else {
                    return 5.76067269e-02f;
                }
            } else {
                if (x[1] < -3.69236171e-02f) {  /* WL */
                    return 1.49721250e-01f;
                } else {
                    return 3.56757641e-02f;
                }
            }
        }
    } else {
        if (x[4] < -9.75628942e-02f) {  /* SNR */
            if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                if (x[5] < 2.84881264e-01f) {  /* LOG */
                    return -3.74064445e-02f;
                } else {
                    return 9.60851181e-03f;
                }
            } else {
                if (x[1] < -3.70229185e-02f) {  /* WL */
                    return 3.94695066e-02f;
                } else {
                    return 8.36094026e-04f;
                }
            }
        } else {
            if (x[14] < -4.96097989e-02f) {  /* MeanFreq */
                if (x[1] < -3.71290185e-02f) {  /* WL */
                    return -7.25145936e-02f;
                } else {
                    return -1.38234468e-02f;
                }
            } else {
                if (x[14] < -4.93106693e-02f) {  /* MeanFreq */
                    return -7.70785660e-02f;
                } else {
                    return 2.96462569e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_82(const float *x) {
    if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
        if (x[13] < 1.33178845e-01f) {  /* SpectralKurtosis */
            if (x[2] < -9.68756974e-01f) {  /* WAMP */
                if (x[4] < -4.00812507e-01f) {  /* SNR */
                    return 6.12204783e-02f;
                } else {
                    return -8.03680569e-02f;
                }
            } else {
                if (x[14] < -5.00449799e-02f) {  /* MeanFreq */
                    return -1.04940310e-02f;
                } else {
                    return 4.72627347e-03f;
                }
            }
        } else {
            if (x[4] < 8.48652959e-01f) {  /* SNR */
                if (x[2] < -7.28667319e-01f) {  /* WAMP */
                    return 2.97350939e-02f;
                } else {
                    return 1.09019831e-01f;
                }
            } else {
                if (x[14] < -5.04316613e-02f) {  /* MeanFreq */
                    return 5.12526091e-03f;
                } else {
                    return -3.10437437e-02f;
                }
            }
        }
    } else {
        if (x[7] < -3.51478942e-02f) {  /* TFR_MeanEnergy */
            if (x[14] < -4.02907543e-02f) {  /* MeanFreq */
                if (x[10] < -4.87960242e-02f) {  /* R_Mid */
                    return -1.87855400e-02f;
                } else {
                    return -7.11139515e-02f;
                }
            } else {
                return 7.59832039e-02f;
            }
        } else {
            if (x[5] < -4.25559103e-01f) {  /* LOG */
                if (x[4] < 5.01608133e-01f) {  /* SNR */
                    return 8.76878854e-03f;
                } else {
                    return 7.78193995e-02f;
                }
            } else {
                if (x[10] < -4.79534864e-02f) {  /* R_Mid */
                    return -6.91399584e-03f;
                } else {
                    return 9.85700451e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_83(const float *x) {
    if (x[5] < -3.40174288e-01f) {  /* LOG */
        if (x[2] < 3.38397712e-01f) {  /* WAMP */
            if (x[10] < -4.89141010e-02f) {  /* R_Mid */
                if (x[2] < -9.42080319e-01f) {  /* WAMP */
                    return -4.67648581e-02f;
                } else {
                    return -1.30892894e-03f;
                }
            } else {
                if (x[13] < -6.28083348e-01f) {  /* SpectralKurtosis */
                    return 6.73210695e-02f;
                } else {
                    return -5.00636511e-02f;
                }
            }
        } else {
            if (x[5] < -4.37254786e-01f) {  /* LOG */
                if (x[13] < -5.51105559e-01f) {  /* SpectralKurtosis */
                    return 1.32157821e-02f;
                } else {
                    return 8.21090862e-02f;
                }
            } else {
                if (x[7] < -3.51421535e-02f) {  /* TFR_MeanEnergy */
                    return -1.54517069e-01f;
                } else {
                    return -9.46604833e-03f;
                }
            }
        }
    } else {
        if (x[7] < -3.51414718e-02f) {  /* TFR_MeanEnergy */
            if (x[7] < -3.51415239e-02f) {  /* TFR_MeanEnergy */
                if (x[0] < -4.65951823e-02f) {  /* MAV */
                    return -5.34264930e-02f;
                } else {
                    return 4.93141972e-02f;
                }
            } else {
                if (x[10] < -5.05954102e-02f) {  /* R_Mid */
                    return 3.73281203e-02f;
                } else {
                    return 1.45142525e-01f;
                }
            }
        } else {
            if (x[10] < -5.07556573e-02f) {  /* R_Mid */
                if (x[2] < -7.55343974e-01f) {  /* WAMP */
                    return -4.68865363e-03f;
                } else {
                    return -5.67190237e-02f;
                }
            } else {
                if (x[13] < 4.17913109e-01f) {  /* SpectralKurtosis */
                    return -5.26365475e-04f;
                } else {
                    return 2.01967359e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_84(const float *x) {
    if (x[9] < -3.62521969e-02f) {  /* E_Low */
        if (x[14] < -5.04852086e-02f) {  /* MeanFreq */
            if (x[1] < -3.74037959e-02f) {  /* WL */
                if (x[4] < 1.82875657e+00f) {  /* SNR */
                    return 3.60046402e-02f;
                } else {
                    return -3.03142052e-03f;
                }
            } else {
                if (x[2] < -8.88727069e-01f) {  /* WAMP */
                    return 1.47595525e-01f;
                } else {
                    return 5.94194494e-02f;
                }
            }
        } else {
            if (x[12] < -9.69668329e-02f) {  /* SpectralEntropy */
                if (x[13] < 4.53442365e-01f) {  /* SpectralKurtosis */
                    return -6.91451356e-02f;
                } else {
                    return -8.22591688e-03f;
                }
            } else {
                if (x[14] < -5.03773093e-02f) {  /* MeanFreq */
                    return 2.29930785e-02f;
                } else {
                    return 8.01620074e-04f;
                }
            }
        }
    } else {
        if (x[10] < -4.47870754e-02f) {  /* R_Mid */
            if (x[2] < 5.78487337e-01f) {  /* WAMP */
                if (x[2] < 1.82782058e-02f) {  /* WAMP */
                    return -6.98691467e-03f;
                } else {
                    return 1.96737275e-02f;
                }
            } else {
                if (x[12] < -6.67910427e-02f) {  /* SpectralEntropy */
                    return -8.95420536e-02f;
                } else {
                    return -1.35590313e-02f;
                }
            }
        } else {
            if (x[1] < -3.63284983e-02f) {  /* WL */
                if (x[1] < -3.69725898e-02f) {  /* WL */
                    return 6.95544258e-02f;
                } else {
                    return -6.23735674e-02f;
                }
            } else {
                if (x[13] < -5.97470582e-01f) {  /* SpectralKurtosis */
                    return 1.48599017e-02f;
                } else {
                    return 1.20733880e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_85(const float *x) {
    if (x[1] < -3.59068140e-02f) {  /* WL */
        if (x[14] < -4.87403758e-02f) {  /* MeanFreq */
            if (x[10] < -4.78977077e-02f) {  /* R_Mid */
                if (x[5] < -4.81943178e+00f) {  /* LOG */
                    return 8.03015977e-02f;
                } else {
                    return 8.24632414e-04f;
                }
            } else {
                if (x[4] < -3.09520453e-01f) {  /* SNR */
                    return 8.29394385e-02f;
                } else {
                    return 8.76982883e-03f;
                }
            }
        } else {
            if (x[14] < -4.85934764e-02f) {  /* MeanFreq */
                if (x[10] < -4.98727895e-02f) {  /* R_Mid */
                    return 7.09802890e-03f;
                } else {
                    return -6.29493222e-02f;
                }
            } else {
                if (x[13] < -5.27613163e-01f) {  /* SpectralKurtosis */
                    return -6.99711218e-03f;
                } else {
                    return 7.74696246e-02f;
                }
            }
        }
    } else {
        if (x[12] < 4.81761731e-02f) {  /* SpectralEntropy */
            if (x[13] < -6.09025300e-01f) {  /* SpectralKurtosis */
                if (x[10] < -4.82886098e-02f) {  /* R_Mid */
                    return 5.50226308e-03f;
                } else {
                    return 6.89837858e-02f;
                }
            } else {
                if (x[6] < -3.62418666e-02f) {  /* SSI */
                    return -3.62000652e-02f;
                } else {
                    return 4.34803553e-02f;
                }
            }
        } else {
            if (x[1] < -3.53484228e-02f) {  /* WL */
                if (x[13] < -6.09025300e-01f) {  /* SpectralKurtosis */
                    return -5.26744425e-02f;
                } else {
                    return 5.15972190e-02f;
                }
            } else {
                if (x[14] < -4.09943499e-02f) {  /* MeanFreq */
                    return 7.05090985e-02f;
                } else {
                    return -2.51148501e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_86(const float *x) {
    if (x[14] < -5.04904240e-02f) {  /* MeanFreq */
        if (x[7] < -3.51394638e-02f) {  /* TFR_MeanEnergy */
            if (x[10] < -5.08026555e-02f) {  /* R_Mid */
                if (x[1] < -3.74854095e-02f) {  /* WL */
                    return 9.37304571e-02f;
                } else {
                    return 7.26260990e-03f;
                }
            } else {
                if (x[4] < 1.82875657e+00f) {  /* SNR */
                    return 1.76062912e-01f;
                } else {
                    return 2.84580588e-02f;
                }
            }
        } else {
            if (x[1] < -3.74724865e-02f) {  /* WL */
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return -8.93830061e-02f;
                } else {
                    return -1.14461472e-02f;
                }
            } else {
                if (x[10] < -5.08119687e-02f) {  /* R_Mid */
                    return 5.19418009e-02f;
                } else {
                    return -5.50596118e-02f;
                }
            }
        }
    } else {
        if (x[1] < -3.74037959e-02f) {  /* WL */
            if (x[0] < -4.97120693e-02f) {  /* MAV */
                if (x[6] < -3.62480395e-02f) {  /* SSI */
                    return 9.40047950e-03f;
                } else {
                    return 1.16540872e-01f;
                }
            } else {
                if (x[14] < -5.04262149e-02f) {  /* MeanFreq */
                    return -9.99879465e-03f;
                } else {
                    return -6.59820363e-02f;
                }
            }
        } else {
            if (x[5] < -8.97694528e-01f) {  /* LOG */
                if (x[7] < -3.51523831e-02f) {  /* TFR_MeanEnergy */
                    return 8.52921605e-03f;
                } else {
                    return -3.14783454e-02f;
                }
            } else {
                if (x[8] < -3.62539366e-02f) {  /* E_Total */
                    return 9.92624834e-02f;
                } else {
                    return 9.99333453e-04f;
                }
            }
        }
    }
}

static inline float xgb_tree_87(const float *x) {
    if (x[2] < -7.28667319e-01f) {  /* WAMP */
        if (x[3] < -3.62440683e-02f) {  /* VAR */
            if (x[1] < -3.73004153e-02f) {  /* WL */
                if (x[2] < -8.08697224e-01f) {  /* WAMP */
                    return 1.13490643e-03f;
                } else {
                    return -3.24666575e-02f;
                }
            } else {
                if (x[13] < 4.53442365e-01f) {  /* SpectralKurtosis */
                    return 7.58076226e-03f;
                } else {
                    return 8.44787508e-02f;
                }
            }
        } else {
            if (x[10] < -4.26135026e-02f) {  /* R_Mid */
                return -1.16901495e-01f;
            } else {
                if (x[1] < -3.68909724e-02f) {  /* WL */
                    return 5.79701848e-02f;
                } else {
                    return -6.81825429e-02f;
                }
            }
        }
    } else {
        if (x[12] < -9.42851081e-02f) {  /* SpectralEntropy */
            if (x[0] < -4.63906750e-02f) {  /* MAV */
                if (x[13] < 1.59917787e-01f) {  /* SpectralKurtosis */
                    return -7.51930242e-03f;
                } else {
                    return 1.46249443e-01f;
                }
            } else {
                if (x[7] < -3.51383314e-02f) {  /* TFR_MeanEnergy */
                    return -6.66796789e-02f;
                } else {
                    return 1.62295941e-02f;
                }
            }
        } else {
            if (x[13] < 2.61404872e-01f) {  /* SpectralKurtosis */
                if (x[3] < -3.62440757e-02f) {  /* VAR */
                    return -6.33934839e-03f;
                } else {
                    return 3.64864245e-03f;
                }
            } else {
                if (x[1] < -3.72582451e-02f) {  /* WL */
                    return 3.10779195e-02f;
                } else {
                    return 1.98677793e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_88(const float *x) {
    if (x[12] < -5.81776053e-02f) {  /* SpectralEntropy */
        if (x[2] < 8.45253587e-01f) {  /* WAMP */
            if (x[8] < -3.62343453e-02f) {  /* E_Total */
                if (x[2] < 3.38397712e-01f) {  /* WAMP */
                    return -2.83204066e-03f;
                } else {
                    return 1.87675282e-02f;
                }
            } else {
                if (x[14] < -4.88003269e-02f) {  /* MeanFreq */
                    return 4.92039621e-02f;
                } else {
                    return -1.10380448e-01f;
                }
            }
        } else {
            if (x[8] < -3.62382941e-02f) {  /* E_Total */
                if (x[6] < -3.62444744e-02f) {  /* SSI */
                    return 3.71394530e-02f;
                } else {
                    return -9.45144072e-02f;
                }
            } else {
                if (x[14] < -4.82949279e-02f) {  /* MeanFreq */
                    return -4.46918420e-02f;
                } else {
                    return 1.75496131e-01f;
                }
            }
        }
    } else {
        if (x[2] < 8.71930242e-01f) {  /* WAMP */
            if (x[3] < -3.62440124e-02f) {  /* VAR */
                if (x[10] < -5.02402373e-02f) {  /* R_Mid */
                    return 6.73908442e-02f;
                } else {
                    return 7.57944677e-03f;
                }
            } else {
                if (x[3] < -3.62438448e-02f) {  /* VAR */
                    return -4.68974784e-02f;
                } else {
                    return 2.43452098e-02f;
                }
            }
        } else {
            if (x[2] < 1.43213940e+00f) {  /* WAMP */
                if (x[3] < -3.62435095e-02f) {  /* VAR */
                    return 3.51919383e-02f;
                } else {
                    return -9.93582681e-02f;
                }
            } else {
                if (x[12] < -4.76557165e-02f) {  /* SpectralEntropy */
                    return -6.53195605e-02f;
                } else {
                    return 7.02456897e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_89(const float *x) {
    if (x[5] < -1.04050420e-01f) {  /* LOG */
        if (x[13] < 3.63706279e+00f) {  /* SpectralKurtosis */
            if (x[4] < 2.37086248e+00f) {  /* SNR */
                if (x[10] < -5.07961437e-02f) {  /* R_Mid */
                    return -3.48421894e-02f;
                } else {
                    return -3.90036078e-03f;
                }
            } else {
                if (x[2] < -8.88727069e-01f) {  /* WAMP */
                    return 1.61151975e-01f;
                } else {
                    return -5.80712818e-02f;
                }
            }
        } else {
            if (x[4] < 2.18982935e+00f) {  /* SNR */
                if (x[4] < 1.44541502e+00f) {  /* SNR */
                    return 2.86639389e-02f;
                } else {
                    return 1.28091246e-01f;
                }
            } else {
                if (x[10] < -5.08396998e-02f) {  /* R_Mid */
                    return 9.72708240e-02f;
                } else {
                    return -8.21478143e-02f;
                }
            }
        }
    } else {
        if (x[5] < -6.99923411e-02f) {  /* LOG */
            if (x[2] < -8.35373819e-01f) {  /* WAMP */
                if (x[13] < 2.32244062e+00f) {  /* SpectralKurtosis */
                    return 4.47169505e-02f;
                } else {
                    return -2.49459092e-02f;
                }
            } else {
                if (x[12] < -9.35497954e-02f) {  /* SpectralEntropy */
                    return -6.08006977e-02f;
                } else {
                    return 2.75595542e-02f;
                }
            }
        } else {
            if (x[12] < 2.44071637e-03f) {  /* SpectralEntropy */
                if (x[2] < 1.59219909e+00f) {  /* WAMP */
                    return -5.72347606e-04f;
                } else {
                    return -6.02988265e-02f;
                }
            } else {
                if (x[3] < -3.62434052e-02f) {  /* VAR */
                    return 2.95041539e-02f;
                } else {
                    return -3.34373824e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_90(const float *x) {
    if (x[7] < -3.51093002e-02f) {  /* TFR_MeanEnergy */
        if (x[11] < 7.94922352e-01f) {  /* TopFreq2 */
            if (x[10] < -5.08336388e-02f) {  /* R_Mid */
                if (x[12] < -9.89609808e-02f) {  /* SpectralEntropy */
                    return -5.89929568e-03f;
                } else {
                    return 1.00554742e-01f;
                }
            } else {
                if (x[10] < -5.08140810e-02f) {  /* R_Mid */
                    return -2.00449713e-02f;
                } else {
                    return 1.35439949e-03f;
                }
            }
        } else {
            return -9.96592566e-02f;
        }
    } else {
        if (x[5] < 7.59880185e-01f) {  /* LOG */
            if (x[2] < -5.68607569e-01f) {  /* WAMP */
                return -6.04108386e-02f;
            } else {
                return 9.82001945e-02f;
            }
        } else {
            if (x[3] < -3.62441055e-02f) {  /* VAR */
                if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                    return 4.39180806e-02f;
                } else {
                    return -5.63467219e-02f;
                }
            } else {
                if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                    return -9.39327627e-02f;
                } else {
                    return -1.86778512e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_91(const float *x) {
    if (x[10] < -4.78977077e-02f) {  /* R_Mid */
        if (x[2] < 2.58367836e-01f) {  /* WAMP */
            if (x[13] < -4.79586214e-01f) {  /* SpectralKurtosis */
                if (x[14] < -4.92841415e-02f) {  /* MeanFreq */
                    return 2.70615872e-02f;
                } else {
                    return 5.91852760e-04f;
                }
            } else {
                if (x[1] < -3.69236171e-02f) {  /* WL */
                    return 7.63717399e-04f;
                } else {
                    return -2.84790844e-02f;
                }
            }
        } else {
            if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                if (x[12] < -5.94669618e-02f) {  /* SpectralEntropy */
                    return 3.66086774e-02f;
                } else {
                    return -8.07283521e-02f;
                }
            } else {
                if (x[12] < -6.88733160e-02f) {  /* SpectralEntropy */
                    return -4.59177271e-02f;
                } else {
                    return -3.66898719e-04f;
                }
            }
        }
    } else {
        if (x[2] < 1.59219909e+00f) {  /* WAMP */
            if (x[5] < -6.27905130e-01f) {  /* LOG */
                if (x[8] < -3.62604596e-02f) {  /* E_Total */
                    return 1.73721369e-02f;
                } else {
                    return -4.89777140e-02f;
                }
            } else {
                if (x[2] < -6.17516711e-02f) {  /* WAMP */
                    return -9.96661279e-03f;
                } else {
                    return 2.96363123e-02f;
                }
            }
        } else {
            if (x[2] < 1.64555240e+00f) {  /* WAMP */
                if (x[12] < 5.69183715e-02f) {  /* SpectralEntropy */
                    return -2.75646131e-02f;
                } else {
                    return -1.62407279e-01f;
                }
            } else {
                if (x[12] < -3.39983730e-03f) {  /* SpectralEntropy */
                    return -5.08503877e-02f;
                } else {
                    return 1.33204479e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_92(const float *x) {
    if (x[12] < -8.11554939e-02f) {  /* SpectralEntropy */
        if (x[1] < -3.69515046e-02f) {  /* WL */
            if (x[2] < -3.28517944e-01f) {  /* WAMP */
                if (x[6] < -3.62545662e-02f) {  /* SSI */
                    return 4.78046052e-02f;
                } else {
                    return -1.45931751e-03f;
                }
            } else {
                if (x[12] < -8.18120092e-02f) {  /* SpectralEntropy */
                    return 4.14431803e-02f;
                } else {
                    return -3.08003407e-02f;
                }
            }
        } else {
            if (x[0] < -3.00452393e-02f) {  /* MAV */
                if (x[0] < -3.42949331e-02f) {  /* MAV */
                    return -5.68274371e-02f;
                } else {
                    return 4.59373072e-02f;
                }
            } else {
                if (x[6] < -3.62375416e-02f) {  /* SSI */
                    return -1.09804347e-01f;
                } else {
                    return -1.94701739e-02f;
                }
            }
        }
    } else {
        if (x[13] < -2.54293323e-01f) {  /* SpectralKurtosis */
            if (x[1] < -3.73480208e-02f) {  /* WL */
                if (x[4] < -4.00812507e-01f) {  /* SNR */
                    return 1.49193719e-01f;
                } else {
                    return -3.14319856e-03f;
                }
            } else {
                if (x[1] < -3.71929519e-02f) {  /* WL */
                    return -9.41203460e-02f;
                } else {
                    return 3.92472837e-03f;
                }
            }
        } else {
            if (x[0] < -4.53202799e-02f) {  /* MAV */
                if (x[1] < -3.72276381e-02f) {  /* WL */
                    return -4.23673652e-02f;
                } else {
                    return 6.45111799e-02f;
                }
            } else {
                if (x[1] < -3.70290391e-02f) {  /* WL */
                    return 1.46581575e-01f;
                } else {
                    return 4.05407213e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_93(const float *x) {
    if (x[13] < 8.70487034e-01f) {  /* SpectralKurtosis */
        if (x[10] < -5.07584922e-02f) {  /* R_Mid */
            if (x[13] < -6.02064431e-02f) {  /* SpectralKurtosis */
                if (x[14] < -5.01016416e-02f) {  /* MeanFreq */
                    return 1.30292073e-01f;
                } else {
                    return -1.34830317e-02f;
                }
            } else {
                if (x[2] < -7.82020569e-01f) {  /* WAMP */
                    return -1.40233785e-02f;
                } else {
                    return -5.65579347e-02f;
                }
            }
        } else {
            if (x[13] < 8.40160906e-01f) {  /* SpectralKurtosis */
                if (x[13] < 3.64703685e-01f) {  /* SpectralKurtosis */
                    return -2.38605379e-03f;
                } else {
                    return 1.52445501e-02f;
                }
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 3.30255739e-02f;
                } else {
                    return -1.17376499e-01f;
                }
            }
        }
    } else {
        if (x[1] < -3.73527855e-02f) {  /* WL */
            if (x[1] < -3.74309979e-02f) {  /* WL */
                if (x[2] < -9.68756974e-01f) {  /* WAMP */
                    return -4.40380827e-04f;
                } else {
                    return -3.43862884e-02f;
                }
            } else {
                if (x[14] < -5.04878275e-02f) {  /* MeanFreq */
                    return 7.72170499e-02f;
                } else {
                    return 9.70926881e-03f;
                }
            }
        } else {
            if (x[14] < -5.04292250e-02f) {  /* MeanFreq */
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return 1.67108670e-01f;
                } else {
                    return 1.09636260e-03f;
                }
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 8.72270390e-02f;
                } else {
                    return -3.03677227e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_94(const float *x) {
    if (x[13] < -6.23944223e-01f) {  /* SpectralKurtosis */
        if (x[10] < -4.93563786e-02f) {  /* R_Mid */
            if (x[7] < -3.51394638e-02f) {  /* TFR_MeanEnergy */
                if (x[10] < -4.98369560e-02f) {  /* R_Mid */
                    return 8.60933438e-02f;
                } else {
                    return -2.55098529e-02f;
                }
            } else {
                if (x[4] < -1.18951559e+00f) {  /* SNR */
                    return -1.07812643e-01f;
                } else {
                    return -1.40048368e-02f;
                }
            }
        } else {
            if (x[2] < -5.95284164e-01f) {  /* WAMP */
                if (x[13] < -6.29362702e-01f) {  /* SpectralKurtosis */
                    return -6.60210028e-02f;
                } else {
                    return 2.45645549e-02f;
                }
            } else {
                if (x[9] < -3.61263938e-02f) {  /* E_Low */
                    return 3.54508124e-02f;
                } else {
                    return 7.88882598e-02f;
                }
            }
        }
    } else {
        if (x[4] < -1.32085180e+00f) {  /* SNR */
            if (x[4] < -1.34111738e+00f) {  /* SNR */
                if (x[7] < -3.51421535e-02f) {  /* TFR_MeanEnergy */
                    return -6.53117225e-02f;
                } else {
                    return 2.05970812e-03f;
                }
            } else {
                if (x[0] < -4.48421948e-02f) {  /* MAV */
                    return -8.23027454e-03f;
                } else {
                    return -1.54155076e-01f;
                }
            }
        } else {
            if (x[13] < -6.09963298e-01f) {  /* SpectralKurtosis */
                if (x[13] < -6.22344434e-01f) {  /* SpectralKurtosis */
                    return -5.36041558e-02f;
                } else {
                    return 3.06425095e-02f;
                }
            } else {
                if (x[0] < 4.67124060e-02f) {  /* MAV */
                    return 8.48847849e-04f;
                } else {
                    return -7.81340897e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_95(const float *x) {
    if (x[1] < -3.73126566e-02f) {  /* WL */
        if (x[2] < -8.35373819e-01f) {  /* WAMP */
            if (x[4] < -4.18776602e-01f) {  /* SNR */
                if (x[1] < -3.74554843e-02f) {  /* WL */
                    return 2.70878226e-02f;
                } else {
                    return 1.21930800e-01f;
                }
            } else {
                if (x[4] < 4.33944255e-01f) {  /* SNR */
                    return -5.04651070e-02f;
                } else {
                    return -1.55116641e-03f;
                }
            }
        } else {
            if (x[5] < -2.37783998e-01f) {  /* LOG */
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 9.90254283e-02f;
                } else {
                    return -6.77901506e-03f;
                }
            } else {
                if (x[1] < -3.73507440e-02f) {  /* WL */
                    return -7.50219151e-02f;
                } else {
                    return -2.72559281e-02f;
                }
            }
        }
    } else {
        if (x[13] < 4.53442365e-01f) {  /* SpectralKurtosis */
            if (x[4] < 1.05037916e+00f) {  /* SNR */
                if (x[14] < -5.03358580e-02f) {  /* MeanFreq */
                    return 2.74415407e-02f;
                } else {
                    return 1.25556195e-03f;
                }
            } else {
                if (x[14] < -5.03773093e-02f) {  /* MeanFreq */
                    return -8.64948109e-02f;
                } else {
                    return 3.70344929e-02f;
                }
            }
        } else {
            if (x[5] < -5.33173643e-02f) {  /* LOG */
                if (x[10] < -5.07584922e-02f) {  /* R_Mid */
                    return -3.71411368e-02f;
                } else {
                    return 4.05316539e-02f;
                }
            } else {
                if (x[4] < 7.71526098e-01f) {  /* SNR */
                    return 2.77183540e-02f;
                } else {
                    return 1.36835441e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_96(const float *x) {
    if (x[5] < -4.81943178e+00f) {  /* LOG */
        if (x[1] < -3.74425612e-02f) {  /* WL */
            if (x[10] < -4.73016351e-02f) {  /* R_Mid */
                return -7.29691759e-02f;
            } else {
                return 4.56222259e-02f;
            }
        } else {
            if (x[4] < -9.96244848e-01f) {  /* SNR */
                if (x[3] < -3.62438895e-02f) {  /* VAR */
                    return -4.23688702e-02f;
                } else {
                    return 5.10616563e-02f;
                }
            } else {
                if (x[13] < -3.95395875e-01f) {  /* SpectralKurtosis */
                    return 1.07373178e-01f;
                } else {
                    return 2.53323372e-02f;
                }
            }
        }
    } else {
        if (x[13] < 3.64703685e-01f) {  /* SpectralKurtosis */
            if (x[12] < -9.44075808e-02f) {  /* SpectralEntropy */
                if (x[10] < -5.06316200e-02f) {  /* R_Mid */
                    return -6.08506463e-02f;
                } else {
                    return -3.41869635e-03f;
                }
            } else {
                if (x[4] < 6.05665863e-01f) {  /* SNR */
                    return -1.61744619e-03f;
                } else {
                    return 4.50560935e-02f;
                }
            }
        } else {
            if (x[1] < -3.73126566e-02f) {  /* WL */
                if (x[5] < -6.08937889e-02f) {  /* LOG */
                    return 8.70137289e-03f;
                } else {
                    return -1.09057743e-02f;
                }
            } else {
                if (x[7] < -3.51393186e-02f) {  /* TFR_MeanEnergy */
                    return -1.29535468e-02f;
                } else {
                    return 7.47982189e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_97(const float *x) {
    if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
        if (x[5] < 2.84881264e-01f) {  /* LOG */
            if (x[7] < -3.51383947e-02f) {  /* TFR_MeanEnergy */
                if (x[5] < -3.89046133e-01f) {  /* LOG */
                    return -8.16446915e-03f;
                } else {
                    return 8.03526398e-03f;
                }
            } else {
                if (x[14] < -4.96292524e-02f) {  /* MeanFreq */
                    return -3.02367681e-03f;
                } else {
                    return -4.11019064e-02f;
                }
            }
        } else {
            if (x[1] < -3.67733091e-02f) {  /* WL */
                if (x[10] < -5.06074429e-02f) {  /* R_Mid */
                    return -2.27519386e-02f;
                } else {
                    return 5.92677668e-02f;
                }
            } else {
                if (x[8] < -3.62393446e-02f) {  /* E_Total */
                    return 5.39025962e-02f;
                } else {
                    return -1.97537392e-02f;
                }
            }
        }
    } else {
        if (x[5] < 1.58213675e+00f) {  /* LOG */
            if (x[11] < 5.80264390e-01f) {  /* TopFreq2 */
                if (x[7] < -3.51267308e-02f) {  /* TFR_MeanEnergy */
                    return -5.12408232e-03f;
                } else {
                    return -6.20047934e-02f;
                }
            } else {
                if (x[10] < -5.04655950e-02f) {  /* R_Mid */
                    return -2.54548136e-02f;
                } else {
                    return 8.28728303e-02f;
                }
            }
        } else {
            if (x[8] < -3.62228528e-02f) {  /* E_Total */
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return 3.86716984e-02f;
                } else {
                    return 2.44242221e-01f;
                }
            } else {
                if (x[8] < -3.61802876e-02f) {  /* E_Total */
                    return 3.61495130e-02f;
                } else {
                    return -8.28403383e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_98(const float *x) {
    if (x[12] < -9.90521535e-02f) {  /* SpectralEntropy */
        if (x[4] < 2.18982935e+00f) {  /* SNR */
            if (x[2] < -9.95433569e-01f) {  /* WAMP */
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return -5.37857115e-02f;
                } else {
                    return 9.39327776e-02f;
                }
            } else {
                if (x[13] < 9.11970377e-01f) {  /* SpectralKurtosis */
                    return -1.39329946e-02f;
                } else {
                    return -8.55840817e-02f;
                }
            }
        } else {
            if (x[0] < -3.97146493e-02f) {  /* MAV */
                if (x[0] < -4.44105752e-02f) {  /* MAV */
                    return -8.40642601e-02f;
                } else {
                    return 5.49620092e-02f;
                }
            } else {
                if (x[0] < -3.77969630e-02f) {  /* MAV */
                    return -1.85392778e-02f;
                } else {
                    return -8.74800906e-02f;
                }
            }
        }
    } else {
        if (x[14] < -5.05044200e-02f) {  /* MeanFreq */
            return 1.16463244e-01f;
        } else {
            if (x[13] < -6.26930475e-01f) {  /* SpectralKurtosis */
                if (x[13] < -6.29870594e-01f) {  /* SpectralKurtosis */
                    return 1.39181165e-03f;
                } else {
                    return 5.73710799e-02f;
                }
            } else {
                if (x[4] < -1.61220288e+00f) {  /* SNR */
                    return -6.45334721e-02f;
                } else {
                    return -1.38816016e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_99(const float *x) {
    if (x[7] < -3.51381525e-02f) {  /* TFR_MeanEnergy */
        if (x[10] < -4.98902127e-02f) {  /* R_Mid */
            if (x[10] < -4.99594212e-02f) {  /* R_Mid */
                if (x[4] < 2.18982935e+00f) {  /* SNR */
                    return 1.55529779e-05f;
                } else {
                    return 3.02735269e-02f;
                }
            } else {
                if (x[4] < -4.08575267e-01f) {  /* SNR */
                    return -9.94094536e-02f;
                } else {
                    return -1.44460276e-02f;
                }
            }
        } else {
            if (x[7] < -3.51421535e-02f) {  /* TFR_MeanEnergy */
                if (x[10] < -4.96475846e-02f) {  /* R_Mid */
                    return 4.88148704e-02f;
                } else {
                    return -1.14508439e-02f;
                }
            } else {
                if (x[4] < -1.08698773e+00f) {  /* SNR */
                    return -2.24313587e-02f;
                } else {
                    return 2.04862002e-02f;
                }
            }
        }
    } else {
        if (x[6] < -3.62442546e-02f) {  /* SSI */
            if (x[2] < 9.83080864e-02f) {  /* WAMP */
                return -1.56178176e-01f;
            } else {
                if (x[7] < -3.51380371e-02f) {  /* TFR_MeanEnergy */
                    return 7.26851076e-02f;
                } else {
                    return -1.22838691e-01f;
                }
            }
        } else {
            if (x[4] < 1.62340498e+00f) {  /* SNR */
                if (x[10] < -5.07870540e-02f) {  /* R_Mid */
                    return 7.05483481e-02f;
                } else {
                    return -3.77236190e-03f;
                }
            } else {
                return -8.12922120e-02f;
            }
        }
    }
}

static inline float xgb_tree_100(const float *x) {
    if (x[7] < -3.51267308e-02f) {  /* TFR_MeanEnergy */
        if (x[12] < -7.12123960e-02f) {  /* SpectralEntropy */
            if (x[2] < 6.85193837e-01f) {  /* WAMP */
                if (x[13] < -5.44014931e-01f) {  /* SpectralKurtosis */
                    return 3.99427637e-02f;
                } else {
                    return -1.88527780e-03f;
                }
            } else {
                if (x[13] < -5.71527719e-01f) {  /* SpectralKurtosis */
                    return -9.73560847e-03f;
                } else {
                    return -1.11655869e-01f;
                }
            }
        } else {
            if (x[13] < -4.75236863e-01f) {  /* SpectralKurtosis */
                if (x[6] < -3.62432674e-02f) {  /* SSI */
                    return -3.08196875e-03f;
                } else {
                    return 1.29205361e-02f;
                }
            } else {
                if (x[2] < -4.08547819e-01f) {  /* WAMP */
                    return -7.23888585e-03f;
                } else {
                    return 7.77810365e-02f;
                }
            }
        }
    } else {
        if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
            if (x[4] < 1.05120815e-01f) {  /* SNR */
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 4.24819620e-04f;
                } else {
                    return -9.30309519e-02f;
                }
            } else {
                if (x[2] < -8.08697224e-01f) {  /* WAMP */
                    return -4.12384830e-02f;
                } else {
                    return 9.89049524e-02f;
                }
            }
        } else {
            if (x[0] < 4.67124060e-02f) {  /* MAV */
                if (x[13] < -5.51105559e-01f) {  /* SpectralKurtosis */
                    return 1.18941821e-01f;
                } else {
                    return -1.54723972e-02f;
                }
            } else {
                if (x[5] < 7.59880185e-01f) {  /* LOG */
                    return 7.69090354e-02f;
                } else {
                    return -7.16599077e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_101(const float *x) {
    if (x[5] < 4.83148384e+00f) {  /* LOG */
        if (x[13] < -6.26930475e-01f) {  /* SpectralKurtosis */
            if (x[13] < -6.30430877e-01f) {  /* SpectralKurtosis */
                if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                    return 9.51294042e-03f;
                } else {
                    return -8.09118077e-02f;
                }
            } else {
                if (x[12] < 2.81396747e-01f) {  /* SpectralEntropy */
                    return 2.94691417e-02f;
                } else {
                    return 9.20731351e-02f;
                }
            }
        } else {
            if (x[12] < 1.93978921e-01f) {  /* SpectralEntropy */
                if (x[1] < -3.72970141e-02f) {  /* WL */
                    return -4.25150013e-03f;
                } else {
                    return 2.55070324e-03f;
                }
            } else {
                if (x[5] < -4.37254786e-01f) {  /* LOG */
                    return 2.33476888e-02f;
                } else {
                    return -9.66088846e-02f;
                }
            }
        }
    } else {
        return -8.48646238e-02f;
    }
}

static inline float xgb_tree_102(const float *x) {
    if (x[14] < -5.04676141e-02f) {  /* MeanFreq */
        if (x[4] < 1.14327848e+00f) {  /* SNR */
            if (x[4] < 1.09602833e+00f) {  /* SNR */
                if (x[10] < -5.07961437e-02f) {  /* R_Mid */
                    return -4.34595607e-02f;
                } else {
                    return 5.57903722e-02f;
                }
            } else {
                return 1.70866102e-01f;
            }
        } else {
            if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                if (x[7] < -3.51394638e-02f) {  /* TFR_MeanEnergy */
                    return 2.16061659e-02f;
                } else {
                    return -5.20719104e-02f;
                }
            } else {
                if (x[7] < -3.51369157e-02f) {  /* TFR_MeanEnergy */
                    return 2.47884486e-02f;
                } else {
                    return -8.29194039e-02f;
                }
            }
        }
    } else {
        if (x[1] < -3.40786092e-02f) {  /* WL */
            if (x[3] < -3.62434611e-02f) {  /* VAR */
                if (x[1] < -3.73126566e-02f) {  /* WL */
                    return -7.64610851e-03f;
                } else {
                    return 9.97337862e-04f;
                }
            } else {
                if (x[3] < -3.62424217e-02f) {  /* VAR */
                    return -6.71428069e-02f;
                } else {
                    return 7.62058869e-02f;
                }
            }
        } else {
            if (x[5] < 1.33196819e+00f) {  /* LOG */
                if (x[5] < 2.72620797e-01f) {  /* LOG */
                    return 9.03156325e-02f;
                } else {
                    return 2.91422810e-02f;
                }
            } else {
                return -4.69903909e-02f;
            }
        }
    }
}

static inline float xgb_tree_103(const float *x) {
    if (x[1] < -3.69236171e-02f) {  /* WL */
        if (x[14] < -4.90190275e-02f) {  /* MeanFreq */
            if (x[13] < -4.83052194e-01f) {  /* SpectralKurtosis */
                if (x[10] < -4.81613986e-02f) {  /* R_Mid */
                    return 5.27819209e-02f;
                } else {
                    return -6.68119034e-03f;
                }
            } else {
                if (x[14] < -4.96097989e-02f) {  /* MeanFreq */
                    return 2.50309752e-03f;
                } else {
                    return -3.12615596e-02f;
                }
            }
        } else {
            if (x[2] < -8.88727069e-01f) {  /* WAMP */
                if (x[1] < -3.73888314e-02f) {  /* WL */
                    return 9.66080204e-02f;
                } else {
                    return 2.60090511e-02f;
                }
            } else {
                if (x[0] < -3.11116558e-02f) {  /* MAV */
                    return -7.67240673e-02f;
                } else {
                    return 3.45274364e-03f;
                }
            }
        }
    } else {
        if (x[1] < -3.69174965e-02f) {  /* WL */
            if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                return -1.93288013e-01f;
            } else {
                if (x[3] < -3.62440832e-02f) {  /* VAR */
                    return -1.03998266e-01f;
                } else {
                    return 8.29224214e-02f;
                }
            }
        } else {
            if (x[10] < -4.66369651e-02f) {  /* R_Mid */
                if (x[7] < -3.51350978e-02f) {  /* TFR_MeanEnergy */
                    return -4.31910390e-03f;
                } else {
                    return -2.13452801e-02f;
                }
            } else {
                if (x[1] < -3.67522240e-02f) {  /* WL */
                    return -4.27194834e-02f;
                } else {
                    return 9.45055392e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_104(const float *x) {
    if (x[5] < 6.83361828e-01f) {  /* LOG */
        if (x[1] < -3.36644053e-02f) {  /* WL */
            if (x[3] < -3.62435095e-02f) {  /* VAR */
                if (x[4] < -1.47969937e+00f) {  /* SNR */
                    return 4.38416265e-02f;
                } else {
                    return 1.42830284e-03f;
                }
            } else {
                if (x[4] < -1.69385076e+00f) {  /* SNR */
                    return -1.27024963e-04f;
                } else {
                    return -1.13194697e-01f;
                }
            }
        } else {
            if (x[2] < 1.13869655e+00f) {  /* WAMP */
                return -6.82736794e-03f;
            } else {
                return 8.93730447e-02f;
            }
        }
    } else {
        if (x[10] < -4.41158749e-02f) {  /* R_Mid */
            if (x[2] < 1.35210955e+00f) {  /* WAMP */
                if (x[2] < 1.24540305e+00f) {  /* WAMP */
                    return -1.95928328e-02f;
                } else {
                    return 1.06287159e-01f;
                }
            } else {
                if (x[1] < -3.53484228e-02f) {  /* WL */
                    return -7.32430071e-02f;
                } else {
                    return 1.35611864e-02f;
                }
            }
        } else {
            if (x[4] < -1.65028071e+00f) {  /* SNR */
                if (x[2] < 4.71780837e-01f) {  /* WAMP */
                    return 3.57768647e-02f;
                } else {
                    return -8.55414495e-02f;
                }
            } else {
                if (x[1] < -3.56959701e-02f) {  /* WL */
                    return 2.87943927e-04f;
                } else {
                    return 9.16277319e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_105(const float *x) {
    if (x[13] < -6.25386894e-01f) {  /* SpectralKurtosis */
        if (x[12] < 4.81761731e-02f) {  /* SpectralEntropy */
            return 8.66912827e-02f;
        } else {
            if (x[4] < -1.69385076e+00f) {  /* SNR */
                if (x[2] < -1.41781554e-01f) {  /* WAMP */
                    return 1.53326793e-04f;
                } else {
                    return 7.32132122e-02f;
                }
            } else {
                if (x[2] < 1.69890559e+00f) {  /* WAMP */
                    return -8.36887136e-02f;
                } else {
                    return 2.61747595e-02f;
                }
            }
        }
    } else {
        if (x[3] < -3.62434052e-02f) {  /* VAR */
            if (x[12] < 1.45064801e-01f) {  /* SpectralEntropy */
                if (x[14] < -4.02907543e-02f) {  /* MeanFreq */
                    return -5.46219235e-04f;
                } else {
                    return -4.20940705e-02f;
                }
            } else {
                if (x[14] < -3.98174636e-02f) {  /* MeanFreq */
                    return -4.90607582e-02f;
                } else {
                    return 7.01852664e-02f;
                }
            }
        } else {
            if (x[7] < -3.51378210e-02f) {  /* TFR_MeanEnergy */
                return 1.21309422e-02f;
            } else {
                if (x[2] < 2.07237840e+00f) {  /* WAMP */
                    return -3.32008712e-02f;
                } else {
                    return -1.13287091e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_106(const float *x) {
    if (x[13] < -6.26930475e-01f) {  /* SpectralKurtosis */
        if (x[2] < -3.81871194e-01f) {  /* WAMP */
            if (x[1] < -3.63727100e-02f) {  /* WL */
                if (x[1] < -3.68828103e-02f) {  /* WL */
                    return 6.67748004e-02f;
                } else {
                    return 2.43577594e-03f;
                }
            } else {
                return -7.64022246e-02f;
            }
        } else {
            if (x[2] < 8.71930242e-01f) {  /* WAMP */
                return 1.25692829e-01f;
            } else {
                if (x[14] < -4.20331433e-02f) {  /* MeanFreq */
                    return 8.23578909e-02f;
                } else {
                    return 1.77945308e-02f;
                }
            }
        }
    } else {
        if (x[14] < -2.89480388e-02f) {  /* MeanFreq */
            if (x[3] < -3.62431556e-02f) {  /* VAR */
                if (x[1] < -3.50926928e-02f) {  /* WL */
                    return -2.80996202e-04f;
                } else {
                    return 3.99753004e-02f;
                }
            } else {
                return -1.14868656e-01f;
            }
        } else {
            if (x[1] < -3.40786092e-02f) {  /* WL */
                if (x[8] < -3.62429023e-02f) {  /* E_Total */
                    return -2.15136081e-01f;
                } else {
                    return -6.32568449e-02f;
                }
            } else {
                if (x[5] < 2.39238739e-01f) {  /* LOG */
                    return 5.97361736e-02f;
                } else {
                    return -3.44145074e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_107(const float *x) {
    if (x[9] < -3.63977104e-02f) {  /* E_Low */
        if (x[12] < -9.26299021e-02f) {  /* SpectralEntropy */
            if (x[1] < -3.73833887e-02f) {  /* WL */
                return 1.38922334e-01f;
            } else {
                if (x[13] < -4.10244465e-01f) {  /* SpectralKurtosis */
                    return 5.16170934e-02f;
                } else {
                    return -2.82709841e-02f;
                }
            }
        } else {
            if (x[10] < -5.04331663e-02f) {  /* R_Mid */
                if (x[10] < -5.05772270e-02f) {  /* R_Mid */
                    return -1.57795884e-02f;
                } else {
                    return -9.89266261e-02f;
                }
            } else {
                if (x[13] < -4.96839792e-01f) {  /* SpectralKurtosis */
                    return 1.38489297e-02f;
                } else {
                    return 5.66115044e-02f;
                }
            }
        }
    } else {
        if (x[10] < -1.40441740e-02f) {  /* R_Mid */
            if (x[9] < -3.63368504e-02f) {  /* E_Low */
                if (x[13] < -6.12878978e-01f) {  /* SpectralKurtosis */
                    return 5.51368892e-02f;
                } else {
                    return -3.40898260e-02f;
                }
            } else {
                if (x[1] < -3.74309979e-02f) {  /* WL */
                    return -1.39547586e-02f;
                } else {
                    return 2.85214419e-03f;
                }
            }
        } else {
            if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                if (x[2] < -5.95284164e-01f) {  /* WAMP */
                    return -6.79825917e-02f;
                } else {
                    return 4.94410507e-02f;
                }
            } else {
                if (x[2] < 1.96567202e+00f) {  /* WAMP */
                    return -1.08595207e-01f;
                } else {
                    return -1.39258010e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_108(const float *x) {
    if (x[12] < 1.65871251e-02f) {  /* SpectralEntropy */
        if (x[8] < -3.62450182e-02f) {  /* E_Total */
            if (x[1] < -3.62047143e-02f) {  /* WL */
                if (x[2] < -6.21960819e-01f) {  /* WAMP */
                    return 9.25881043e-03f;
                } else {
                    return -2.72642146e-03f;
                }
            } else {
                if (x[12] < 5.50472876e-03f) {  /* SpectralEntropy */
                    return 5.42587042e-02f;
                } else {
                    return -1.49030089e-02f;
                }
            }
        } else {
            if (x[5] < -1.72857579e-03f) {  /* LOG */
                if (x[5] < -4.12173346e-02f) {  /* LOG */
                    return 7.48234913e-02f;
                } else {
                    return -4.99645583e-02f;
                }
            } else {
                if (x[9] < -3.61977406e-02f) {  /* E_Low */
                    return -5.91673423e-03f;
                } else {
                    return 1.21000661e-02f;
                }
            }
        }
    } else {
        if (x[3] < -3.62438783e-02f) {  /* VAR */
            if (x[13] < -6.01742804e-01f) {  /* SpectralKurtosis */
                if (x[14] < -4.46994491e-02f) {  /* MeanFreq */
                    return -1.86560854e-01f;
                } else {
                    return -3.78995202e-02f;
                }
            } else {
                if (x[0] < -9.27947909e-02f) {  /* MAV */
                    return -5.65551110e-02f;
                } else {
                    return 8.78105313e-02f;
                }
            }
        } else {
            if (x[0] < -1.26246616e-01f) {  /* MAV */
                if (x[3] < -3.62427868e-02f) {  /* VAR */
                    return 1.05231263e-01f;
                } else {
                    return 2.15558950e-02f;
                }
            } else {
                if (x[1] < -3.60625647e-02f) {  /* WL */
                    return -5.65735698e-02f;
                } else {
                    return -3.93833313e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_109(const float *x) {
    if (x[13] < -6.23465419e-01f) {  /* SpectralKurtosis */
        if (x[10] < -4.94994484e-02f) {  /* R_Mid */
            if (x[0] < -4.46987413e-02f) {  /* MAV */
                if (x[0] < -5.51570095e-02f) {  /* MAV */
                    return -2.97181937e-03f;
                } else {
                    return 6.35393858e-02f;
                }
            } else {
                if (x[2] < 1.40546274e+00f) {  /* WAMP */
                    return -1.41272724e-01f;
                } else {
                    return -2.73726583e-02f;
                }
            }
        } else {
            if (x[3] < -3.62435095e-02f) {  /* VAR */
                if (x[1] < -3.55742276e-02f) {  /* WL */
                    return 1.75075121e-02f;
                } else {
                    return 7.22051486e-02f;
                }
            } else {
                if (x[14] < -3.60739417e-02f) {  /* MeanFreq */
                    return -6.22463524e-02f;
                } else {
                    return 2.97847036e-02f;
                }
            }
        }
    } else {
        if (x[14] < -4.24311534e-02f) {  /* MeanFreq */
            if (x[13] < -6.09963298e-01f) {  /* SpectralKurtosis */
                if (x[10] < -4.85422835e-02f) {  /* R_Mid */
                    return -1.13452151e-02f;
                } else {
                    return 4.02253494e-02f;
                }
            } else {
                if (x[2] < 1.51216924e+00f) {  /* WAMP */
                    return -5.83920511e-04f;
                } else {
                    return -3.47112902e-02f;
                }
            }
        } else {
            if (x[10] < -4.77673225e-02f) {  /* R_Mid */
                if (x[14] < -3.93500440e-02f) {  /* MeanFreq */
                    return 7.13721514e-02f;
                } else {
                    return -3.73468213e-02f;
                }
            } else {
                if (x[10] < -3.15503068e-02f) {  /* R_Mid */
                    return -6.42137900e-02f;
                } else {
                    return 3.89698939e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_110(const float *x) {
    if (x[14] < -4.96958196e-02f) {  /* MeanFreq */
        if (x[13] < -4.51139510e-01f) {  /* SpectralKurtosis */
            if (x[7] < -3.51370499e-02f) {  /* TFR_MeanEnergy */
                if (x[1] < -3.71684693e-02f) {  /* WL */
                    return 3.48827103e-03f;
                } else {
                    return 7.40588531e-02f;
                }
            } else {
                if (x[13] < -4.59048599e-01f) {  /* SpectralKurtosis */
                    return -6.83874264e-02f;
                } else {
                    return 2.28539407e-02f;
                }
            }
        } else {
            if (x[6] < -3.62451226e-02f) {  /* SSI */
                if (x[14] < -5.05044200e-02f) {  /* MeanFreq */
                    return 7.14730322e-02f;
                } else {
                    return -5.53836394e-03f;
                }
            } else {
                if (x[0] < -4.02923413e-02f) {  /* MAV */
                    return 5.36423735e-02f;
                } else {
                    return 4.66199033e-03f;
                }
            }
        }
    } else {
        if (x[13] < -3.73214781e-01f) {  /* SpectralKurtosis */
            if (x[14] < -4.94047999e-02f) {  /* MeanFreq */
                if (x[0] < -5.61225004e-02f) {  /* MAV */
                    return 3.76929790e-02f;
                } else {
                    return -2.75490563e-02f;
                }
            } else {
                if (x[13] < -4.89954382e-01f) {  /* SpectralKurtosis */
                    return -1.20098889e-03f;
                } else {
                    return 4.79969755e-02f;
                }
            }
        } else {
            if (x[4] < -2.52138376e-01f) {  /* SNR */
                return 1.39727652e-01f;
            } else {
                return -7.40614310e-02f;
            }
        }
    }
}

static inline float xgb_tree_111(const float *x) {
    if (x[14] < -5.04774898e-02f) {  /* MeanFreq */
        if (x[1] < -3.74514051e-02f) {  /* WL */
            if (x[2] < -9.68756974e-01f) {  /* WAMP */
                if (x[0] < -3.94476987e-02f) {  /* MAV */
                    return 4.90577035e-02f;
                } else {
                    return -1.00263497e-02f;
                }
            } else {
                if (x[4] < 1.30301583e+00f) {  /* SNR */
                    return 1.22579254e-01f;
                } else {
                    return -6.45881891e-02f;
                }
            }
        } else {
            if (x[0] < -3.86256613e-02f) {  /* MAV */
                if (x[0] < -3.98832969e-02f) {  /* MAV */
                    return 2.36235540e-02f;
                } else {
                    return 1.37453139e-01f;
                }
            } else {
                if (x[2] < -9.42080319e-01f) {  /* WAMP */
                    return 4.47262563e-02f;
                } else {
                    return -7.24941716e-02f;
                }
            }
        }
    } else {
        if (x[1] < -3.73779461e-02f) {  /* WL */
            if (x[10] < -5.08264527e-02f) {  /* R_Mid */
                if (x[12] < -9.81095955e-02f) {  /* SpectralEntropy */
                    return 1.41435368e-02f;
                } else {
                    return -9.88242254e-02f;
                }
            } else {
                if (x[10] < -4.99250144e-02f) {  /* R_Mid */
                    return -1.65436175e-02f;
                } else {
                    return 4.08336073e-02f;
                }
            }
        } else {
            if (x[14] < -5.04555069e-02f) {  /* MeanFreq */
                if (x[10] < -5.07291779e-02f) {  /* R_Mid */
                    return 7.89374337e-02f;
                } else {
                    return -9.63239372e-02f;
                }
            } else {
                if (x[0] < -7.07823411e-03f) {  /* MAV */
                    return -1.02633971e-03f;
                } else {
                    return 2.14991402e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_112(const float *x) {
    if (x[8] < -3.62528414e-02f) {  /* E_Total */
        if (x[12] < -7.95188099e-02f) {  /* SpectralEntropy */
            if (x[6] < -3.62545662e-02f) {  /* SSI */
                if (x[1] < -3.74554843e-02f) {  /* WL */
                    return 1.25684366e-01f;
                } else {
                    return -7.71046791e-04f;
                }
            } else {
                if (x[12] < -8.28233883e-02f) {  /* SpectralEntropy */
                    return -4.77670059e-02f;
                } else {
                    return -1.26132831e-01f;
                }
            }
        } else {
            if (x[10] < -4.94689420e-02f) {  /* R_Mid */
                if (x[7] < -3.51577438e-02f) {  /* TFR_MeanEnergy */
                    return -5.04064374e-02f;
                } else {
                    return 5.92789166e-02f;
                }
            } else {
                if (x[3] < -3.62439044e-02f) {  /* VAR */
                    return -1.80225819e-02f;
                } else {
                    return 2.51348596e-02f;
                }
            }
        }
    } else {
        if (x[8] < -3.62517126e-02f) {  /* E_Total */
            if (x[10] < -5.06512336e-02f) {  /* R_Mid */
                if (x[8] < -3.62524800e-02f) {  /* E_Total */
                    return -2.40991339e-02f;
                } else {
                    return 1.57052353e-01f;
                }
            } else {
                if (x[2] < -8.08697224e-01f) {  /* WAMP */
                    return -8.35980177e-02f;
                } else {
                    return 3.40970084e-02f;
                }
            }
        } else {
            if (x[4] < -1.08698773e+00f) {  /* SNR */
                if (x[1] < -3.57918739e-02f) {  /* WL */
                    return -5.98803572e-02f;
                } else {
                    return 4.58838278e-03f;
                }
            } else {
                if (x[10] < -4.89473641e-02f) {  /* R_Mid */
                    return -1.03028002e-03f;
                } else {
                    return 1.06237950e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_113(const float *x) {
    if (x[13] < -5.22637427e-01f) {  /* SpectralKurtosis */
        if (x[7] < -3.51423547e-02f) {  /* TFR_MeanEnergy */
            if (x[10] < -5.00742309e-02f) {  /* R_Mid */
                if (x[6] < -3.62482220e-02f) {  /* SSI */
                    return -1.62522774e-02f;
                } else {
                    return -1.30278572e-01f;
                }
            } else {
                if (x[0] < -4.93814088e-02f) {  /* MAV */
                    return -7.86123052e-03f;
                } else {
                    return -1.36436105e-01f;
                }
            }
        } else {
            if (x[6] < -3.62465568e-02f) {  /* SSI */
                if (x[13] < -5.44014931e-01f) {  /* SpectralKurtosis */
                    return 7.78294131e-02f;
                } else {
                    return -4.37764674e-02f;
                }
            } else {
                if (x[14] < -4.91324887e-02f) {  /* MeanFreq */
                    return 2.60811448e-02f;
                } else {
                    return -3.19043221e-03f;
                }
            }
        }
    } else {
        if (x[0] < -5.21450303e-02f) {  /* MAV */
            if (x[2] < -8.88727069e-01f) {  /* WAMP */
                if (x[6] < -3.62480395e-02f) {  /* SSI */
                    return -2.96489932e-02f;
                } else {
                    return 6.98343739e-02f;
                }
            } else {
                if (x[4] < 8.73469055e-01f) {  /* SNR */
                    return 3.44524570e-02f;
                } else {
                    return 1.46161616e-01f;
                }
            }
        } else {
            if (x[7] < -3.51430550e-02f) {  /* TFR_MeanEnergy */
                if (x[4] < 9.03083384e-01f) {  /* SNR */
                    return -9.14112851e-02f;
                } else {
                    return 2.40477175e-02f;
                }
            } else {
                if (x[14] < -4.93106693e-02f) {  /* MeanFreq */
                    return 2.43389001e-03f;
                } else {
                    return 2.93442439e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_114(const float *x) {
    if (x[14] < -5.02355546e-02f) {  /* MeanFreq */
        if (x[10] < -5.03502190e-02f) {  /* R_Mid */
            if (x[2] < -7.01990664e-01f) {  /* WAMP */
                if (x[1] < -3.74663658e-02f) {  /* WL */
                    return -2.06288807e-02f;
                } else {
                    return 6.65800739e-03f;
                }
            } else {
                if (x[0] < -4.63906750e-02f) {  /* MAV */
                    return 1.23190008e-01f;
                } else {
                    return -3.58699225e-02f;
                }
            }
        } else {
            if (x[2] < -8.35373819e-01f) {  /* WAMP */
                if (x[10] < -4.99970391e-02f) {  /* R_Mid */
                    return 9.38877289e-04f;
                } else {
                    return 1.11165695e-01f;
                }
            } else {
                if (x[14] < -5.02666123e-02f) {  /* MeanFreq */
                    return 3.25593166e-02f;
                } else {
                    return 1.33138821e-01f;
                }
            }
        }
    } else {
        if (x[2] < -7.82020569e-01f) {  /* WAMP */
            if (x[10] < -5.05954102e-02f) {  /* R_Mid */
                if (x[4] < 3.08667660e-01f) {  /* SNR */
                    return 2.76647601e-02f;
                } else {
                    return -5.90247400e-02f;
                }
            } else {
                if (x[4] < 6.50155306e-01f) {  /* SNR */
                    return -3.53035145e-02f;
                } else {
                    return 8.87962952e-02f;
                }
            }
        } else {
            if (x[12] < -9.30441320e-02f) {  /* SpectralEntropy */
                if (x[0] < -4.09151763e-02f) {  /* MAV */
                    return -1.30050024e-02f;
                } else {
                    return -1.23270355e-01f;
                }
            } else {
                if (x[1] < -3.71052176e-02f) {  /* WL */
                    return 1.50916250e-02f;
                } else {
                    return -4.12701769e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_115(const float *x) {
    if (x[5] < -8.87488872e-02f) {  /* LOG */
        if (x[13] < 2.61454844e+00f) {  /* SpectralKurtosis */
            if (x[8] < -3.62458527e-02f) {  /* E_Total */
                if (x[2] < 1.05866659e+00f) {  /* WAMP */
                    return -6.62151817e-03f;
                } else {
                    return 2.21537370e-02f;
                }
            } else {
                if (x[4] < -1.80371988e+00f) {  /* SNR */
                    return 6.79573417e-02f;
                } else {
                    return -6.92376494e-02f;
                }
            }
        } else {
            if (x[4] < 1.35618246e+00f) {  /* SNR */
                if (x[10] < -5.08098230e-02f) {  /* R_Mid */
                    return 1.40345367e-02f;
                } else {
                    return -9.96147022e-02f;
                }
            } else {
                if (x[4] < 1.82875657e+00f) {  /* SNR */
                    return 9.81316268e-02f;
                } else {
                    return 1.30974492e-02f;
                }
            }
        }
    } else {
        if (x[5] < -2.03995481e-02f) {  /* LOG */
            if (x[1] < -3.74425612e-02f) {  /* WL */
                if (x[6] < -3.62446681e-02f) {  /* SSI */
                    return -3.05291954e-02f;
                } else {
                    return 1.11209594e-01f;
                }
            } else {
                if (x[13] < 1.60007226e+00f) {  /* SpectralKurtosis */
                    return 1.53904874e-02f;
                } else {
                    return 8.37637410e-02f;
                }
            }
        } else {
            if (x[5] < 9.27969217e-01f) {  /* LOG */
                if (x[10] < -5.07823452e-02f) {  /* R_Mid */
                    return 2.60388013e-02f;
                } else {
                    return -3.43534513e-03f;
                }
            } else {
                if (x[12] < -4.86463681e-02f) {  /* SpectralEntropy */
                    return 4.44699489e-02f;
                } else {
                    return -2.54718098e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_116(const float *x) {
    if (x[2] < -9.95433569e-01f) {  /* WAMP */
        if (x[14] < -5.04240245e-02f) {  /* MeanFreq */
            if (x[14] < -5.04700616e-02f) {  /* MeanFreq */
                if (x[4] < 1.14327848e+00f) {  /* SNR */
                    return 1.81502894e-01f;
                } else {
                    return 1.02407495e-02f;
                }
            } else {
                if (x[12] < -9.62071270e-02f) {  /* SpectralEntropy */
                    return -4.36539054e-02f;
                } else {
                    return 6.28306568e-02f;
                }
            }
        } else {
            if (x[10] < -5.05052879e-02f) {  /* R_Mid */
                if (x[13] < 5.33994794e-01f) {  /* SpectralKurtosis */
                    return 1.32080853e-01f;
                } else {
                    return 2.78610140e-02f;
                }
            } else {
                if (x[4] < 5.45021296e-01f) {  /* SNR */
                    return 9.37035866e-03f;
                } else {
                    return -7.39781037e-02f;
                }
            }
        }
    } else {
        if (x[13] < 2.61454844e+00f) {  /* SpectralKurtosis */
            if (x[14] < -5.05066253e-02f) {  /* MeanFreq */
                return 1.13856018e-01f;
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 4.06993786e-03f;
                } else {
                    return -3.03113065e-03f;
                }
            }
        } else {
            if (x[8] < -3.62456813e-02f) {  /* E_Total */
                if (x[6] < -3.62451784e-02f) {  /* SSI */
                    return -4.09624986e-02f;
                } else {
                    return 4.44864072e-02f;
                }
            } else {
                if (x[0] < -3.80068049e-02f) {  /* MAV */
                    return -9.37002823e-02f;
                } else {
                    return -9.92717128e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_117(const float *x) {
    if (x[4] < -1.32085180e+00f) {  /* SNR */
        if (x[2] < 1.69890559e+00f) {  /* WAMP */
            if (x[10] < -4.67249490e-02f) {  /* R_Mid */
                if (x[2] < 1.45881605e+00f) {  /* WAMP */
                    return -1.44299358e-01f;
                } else {
                    return -4.98361886e-02f;
                }
            } else {
                if (x[2] < 1.43213940e+00f) {  /* WAMP */
                    return 4.25301772e-03f;
                } else {
                    return -6.88874274e-02f;
                }
            }
        } else {
            if (x[8] < -3.62353064e-02f) {  /* E_Total */
                if (x[4] < -1.41058254e+00f) {  /* SNR */
                    return 4.97335680e-02f;
                } else {
                    return -3.59570496e-02f;
                }
            } else {
                if (x[12] < 1.21707410e-01f) {  /* SpectralEntropy */
                    return 5.67882471e-02f;
                } else {
                    return -6.56310841e-02f;
                }
            }
        }
    } else {
        if (x[2] < 1.72558236e+00f) {  /* WAMP */
            if (x[12] < -6.21565394e-02f) {  /* SpectralEntropy */
                if (x[2] < 8.45253587e-01f) {  /* WAMP */
                    return 6.99021348e-06f;
                } else {
                    return -5.78477383e-02f;
                }
            } else {
                if (x[10] < -5.04655950e-02f) {  /* R_Mid */
                    return 7.49139413e-02f;
                } else {
                    return 8.08229204e-03f;
                }
            }
        } else {
            if (x[2] < 2.12573171e+00f) {  /* WAMP */
                if (x[7] < -3.51350978e-02f) {  /* TFR_MeanEnergy */
                    return -7.11206123e-02f;
                } else {
                    return -8.31073709e-03f;
                }
            } else {
                return 7.69673809e-02f;
            }
        }
    }
}

static inline float xgb_tree_118(const float *x) {
    if (x[7] < -3.51402126e-02f) {  /* TFR_MeanEnergy */
        if (x[0] < -4.29019332e-02f) {  /* MAV */
            if (x[8] < -3.62473428e-02f) {  /* E_Total */
                if (x[2] < -9.42080319e-01f) {  /* WAMP */
                    return -2.41625179e-02f;
                } else {
                    return 7.80886214e-04f;
                }
            } else {
                if (x[13] < -3.42437446e-01f) {  /* SpectralKurtosis */
                    return 5.63417636e-02f;
                } else {
                    return 2.05267826e-03f;
                }
            }
        } else {
            if (x[4] < 1.62340498e+00f) {  /* SNR */
                if (x[0] < -4.24397588e-02f) {  /* MAV */
                    return -6.20740354e-02f;
                } else {
                    return -2.90786935e-04f;
                }
            } else {
                if (x[13] < 1.16086781e+00f) {  /* SpectralKurtosis */
                    return 3.89406248e-03f;
                } else {
                    return 1.28966257e-01f;
                }
            }
        }
    } else {
        if (x[7] < -3.51401418e-02f) {  /* TFR_MeanEnergy */
            if (x[10] < -4.64343987e-02f) {  /* R_Mid */
                if (x[10] < -5.07980436e-02f) {  /* R_Mid */
                    return -4.57291901e-02f;
                } else {
                    return 7.14174807e-02f;
                }
            } else {
                return -9.69993770e-02f;
            }
        } else {
            if (x[10] < -4.75467108e-02f) {  /* R_Mid */
                if (x[2] < 7.38547087e-01f) {  /* WAMP */
                    return 3.25668533e-03f;
                } else {
                    return -2.02469509e-02f;
                }
            } else {
                if (x[2] < 1.59219909e+00f) {  /* WAMP */
                    return 2.51708515e-02f;
                } else {
                    return -1.22119030e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_119(const float *x) {
    if (x[10] < -3.68184820e-02f) {  /* R_Mid */
        if (x[10] < -4.10061479e-02f) {  /* R_Mid */
            if (x[10] < -4.35475595e-02f) {  /* R_Mid */
                if (x[10] < -5.06691150e-02f) {  /* R_Mid */
                    return -7.11131608e-03f;
                } else {
                    return 1.03390543e-03f;
                }
            } else {
                if (x[3] < -3.62434611e-02f) {  /* VAR */
                    return -1.94351282e-02f;
                } else {
                    return -1.65745571e-01f;
                }
            }
        } else {
            if (x[1] < -3.66250388e-02f) {  /* WL */
                if (x[0] < -1.66312024e-01f) {  /* MAV */
                    return 8.22015554e-02f;
                } else {
                    return -1.12867631e-01f;
                }
            } else {
                if (x[7] < -3.51429246e-02f) {  /* TFR_MeanEnergy */
                    return -1.57025140e-02f;
                } else {
                    return 6.28190264e-02f;
                }
            }
        }
    } else {
        if (x[12] < -2.20234711e-02f) {  /* SpectralEntropy */
            if (x[7] < -3.51357721e-02f) {  /* TFR_MeanEnergy */
                return 8.90864357e-02f;
            } else {
                return 2.06478573e-02f;
            }
        } else {
            if (x[3] < -3.62439789e-02f) {  /* VAR */
                return -1.60741597e-01f;
            } else {
                if (x[3] < -3.62438969e-02f) {  /* VAR */
                    return 3.18668075e-02f;
                } else {
                    return -3.23345177e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_120(const float *x) {
    if (x[13] < 8.70487034e-01f) {  /* SpectralKurtosis */
        if (x[12] < -9.74687114e-02f) {  /* SpectralEntropy */
            if (x[10] < -5.08236587e-02f) {  /* R_Mid */
                return 1.06892407e-01f;
            } else {
                if (x[2] < -8.62050474e-01f) {  /* WAMP */
                    return 1.40672196e-02f;
                } else {
                    return -4.85570282e-02f;
                }
            }
        } else {
            if (x[13] < 6.97020054e-01f) {  /* SpectralKurtosis */
                if (x[14] < -5.03989160e-02f) {  /* MeanFreq */
                    return 3.22810970e-02f;
                } else {
                    return -1.86254270e-04f;
                }
            } else {
                if (x[2] < -8.62050474e-01f) {  /* WAMP */
                    return -4.27239724e-02f;
                } else {
                    return 3.58067527e-02f;
                }
            }
        }
    } else {
        if (x[12] < -9.89609808e-02f) {  /* SpectralEntropy */
            if (x[6] < -3.62463295e-02f) {  /* SSI */
                if (x[13] < 1.39467204e+00f) {  /* SpectralKurtosis */
                    return -2.80427970e-02f;
                } else {
                    return -1.04820013e-01f;
                }
            } else {
                if (x[0] < -4.22697887e-02f) {  /* MAV */
                    return 5.98415248e-02f;
                } else {
                    return -1.89947449e-02f;
                }
            }
        } else {
            if (x[2] < -7.82020569e-01f) {  /* WAMP */
                if (x[0] < -5.71862347e-02f) {  /* MAV */
                    return -6.35790154e-02f;
                } else {
                    return 1.44440373e-02f;
                }
            } else {
                if (x[0] < -4.04025801e-02f) {  /* MAV */
                    return 2.50482589e-01f;
                } else {
                    return -1.32901585e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_121(const float *x) {
    if (x[2] < 2.28579140e+00f) {  /* WAMP */
        if (x[8] < -3.62495258e-02f) {  /* E_Total */
            if (x[6] < -3.62469777e-02f) {  /* SSI */
                if (x[13] < -5.17354071e-01f) {  /* SpectralKurtosis */
                    return -8.35091714e-03f;
                } else {
                    return 1.50300907e-02f;
                }
            } else {
                if (x[10] < -5.05392402e-02f) {  /* R_Mid */
                    return -3.48689146e-02f;
                } else {
                    return 6.95085377e-02f;
                }
            }
        } else {
            if (x[5] < -3.40174288e-01f) {  /* LOG */
                if (x[4] < 6.38242006e-01f) {  /* SNR */
                    return -1.88495275e-02f;
                } else {
                    return -7.00802281e-02f;
                }
            } else {
                if (x[5] < -2.61942327e-01f) {  /* LOG */
                    return 1.62021741e-02f;
                } else {
                    return -6.34148426e-04f;
                }
            }
        }
    } else {
        if (x[0] < -7.07823411e-03f) {  /* MAV */
            return 8.57598111e-02f;
        } else {
            return 5.72664489e-04f;
        }
    }
}

static inline float xgb_tree_122(const float *x) {
    if (x[11] < 7.94922352e-01f) {  /* TopFreq2 */
        if (x[3] < -3.62436101e-02f) {  /* VAR */
            if (x[13] < -6.18349969e-01f) {  /* SpectralKurtosis */
                if (x[10] < -4.93235514e-02f) {  /* R_Mid */
                    return -3.28293554e-02f;
                } else {
                    return 4.41993363e-02f;
                }
            } else {
                if (x[7] < -3.51286046e-02f) {  /* TFR_MeanEnergy */
                    return -1.47598668e-03f;
                } else {
                    return 2.25934424e-02f;
                }
            }
        } else {
            if (x[6] < -3.62464152e-02f) {  /* SSI */
                if (x[10] < -2.76443101e-02f) {  /* R_Mid */
                    return 7.19323829e-02f;
                } else {
                    return -2.89894617e-03f;
                }
            } else {
                if (x[7] < -3.51402126e-02f) {  /* TFR_MeanEnergy */
                    return -1.54937282e-01f;
                } else {
                    return -2.47847419e-02f;
                }
            }
        }
    } else {
        if (x[5] < 3.56144384e-02f) {  /* LOG */
            return -1.26247212e-01f;
        } else {
            return -1.09144691e-02f;
        }
    }
}

static inline float xgb_tree_123(const float *x) {
    if (x[7] < -3.51437554e-02f) {  /* TFR_MeanEnergy */
        if (x[6] < -3.62486690e-02f) {  /* SSI */
            if (x[12] < -5.88143468e-02f) {  /* SpectralEntropy */
                if (x[14] < -4.92841415e-02f) {  /* MeanFreq */
                    return 9.47366003e-04f;
                } else {
                    return -5.96922003e-02f;
                }
            } else {
                if (x[10] < -5.02868630e-02f) {  /* R_Mid */
                    return 9.95878056e-02f;
                } else {
                    return 1.63187180e-02f;
                }
            }
        } else {
            if (x[10] < -5.07096276e-02f) {  /* R_Mid */
                if (x[0] < -5.71862347e-02f) {  /* MAV */
                    return 1.23576215e-02f;
                } else {
                    return 1.32181168e-01f;
                }
            } else {
                if (x[4] < -3.19111764e-01f) {  /* SNR */
                    return 4.94935885e-02f;
                } else {
                    return -8.36856291e-03f;
                }
            }
        }
    } else {
        if (x[10] < -5.07961437e-02f) {  /* R_Mid */
            if (x[7] < -3.51412632e-02f) {  /* TFR_MeanEnergy */
                if (x[10] < -5.08186333e-02f) {  /* R_Mid */
                    return -7.35758990e-02f;
                } else {
                    return 1.07396893e-01f;
                }
            } else {
                if (x[10] < -5.08072525e-02f) {  /* R_Mid */
                    return -5.44791669e-03f;
                } else {
                    return -5.04977070e-02f;
                }
            }
        } else {
            if (x[14] < -5.04904240e-02f) {  /* MeanFreq */
                if (x[0] < -4.01794463e-02f) {  /* MAV */
                    return 1.57777607e-01f;
                } else {
                    return -8.29740148e-03f;
                }
            } else {
                if (x[10] < -5.07823452e-02f) {  /* R_Mid */
                    return 1.92152467e-02f;
                } else {
                    return -1.27631612e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_124(const float *x) {
    if (x[5] < -1.74581444e+00f) {  /* LOG */
        if (x[4] < 4.91116464e-01f) {  /* SNR */
            if (x[10] < -5.03421463e-02f) {  /* R_Mid */
                if (x[4] < 2.75301337e-01f) {  /* SNR */
                    return -1.02243818e-01f;
                } else {
                    return -2.23349109e-02f;
                }
            } else {
                if (x[8] < -3.62651497e-02f) {  /* E_Total */
                    return 1.78254750e-02f;
                } else {
                    return 8.07381198e-02f;
                }
            }
        } else {
            return 1.15260139e-01f;
        }
    } else {
        if (x[10] < -1.40441740e-02f) {  /* R_Mid */
            if (x[10] < -3.15503068e-02f) {  /* R_Mid */
                if (x[10] < -3.47048715e-02f) {  /* R_Mid */
                    return 1.25641469e-04f;
                } else {
                    return -5.60118519e-02f;
                }
            } else {
                if (x[1] < -3.63863111e-02f) {  /* WL */
                    return -6.13426231e-02f;
                } else {
                    return 6.69341683e-02f;
                }
            }
        } else {
            if (x[7] < -3.51378210e-02f) {  /* TFR_MeanEnergy */
                if (x[4] < -1.54304183e+00f) {  /* SNR */
                    return -1.78130925e-01f;
                } else {
                    return -5.55897653e-02f;
                }
            } else {
                if (x[5] < 6.30395338e-02f) {  /* LOG */
                    return 6.48545399e-02f;
                } else {
                    return -2.68056449e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_125(const float *x) {
    if (x[10] < -5.05392402e-02f) {  /* R_Mid */
        if (x[10] < -5.05772270e-02f) {  /* R_Mid */
            if (x[12] < -7.56563991e-02f) {  /* SpectralEntropy */
                if (x[2] < -7.28667319e-01f) {  /* WAMP */
                    return 1.11994578e-03f;
                } else {
                    return -1.83170531e-02f;
                }
            } else {
                if (x[1] < -3.69236171e-02f) {  /* WL */
                    return 1.09524928e-01f;
                } else {
                    return 8.21969844e-03f;
                }
            }
        } else {
            if (x[5] < 2.05824599e-02f) {  /* LOG */
                if (x[2] < -9.95433569e-01f) {  /* WAMP */
                    return 6.65878281e-02f;
                } else {
                    return -5.47942780e-02f;
                }
            } else {
                if (x[1] < -3.72385234e-02f) {  /* WL */
                    return -3.17009129e-02f;
                } else {
                    return 5.95275760e-02f;
                }
            }
        }
    } else {
        if (x[1] < -3.72915752e-02f) {  /* WL */
            if (x[4] < 7.11991787e-01f) {  /* SNR */
                if (x[12] < -9.47066844e-02f) {  /* SpectralEntropy */
                    return -8.72015283e-02f;
                } else {
                    return -1.83764007e-02f;
                }
            } else {
                if (x[10] < -5.04248068e-02f) {  /* R_Mid */
                    return -3.44542973e-02f;
                } else {
                    return 6.17003739e-02f;
                }
            }
        } else {
            if (x[14] < -5.02009764e-02f) {  /* MeanFreq */
                if (x[1] < -3.72419208e-02f) {  /* WL */
                    return 5.70043810e-02f;
                } else {
                    return 2.73020822e-03f;
                }
            } else {
                if (x[1] < -3.72317173e-02f) {  /* WL */
                    return -3.90092432e-02f;
                } else {
                    return 4.54052957e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_126(const float *x) {
    if (x[10] < -5.08435890e-02f) {  /* R_Mid */
        if (x[0] < -3.97146493e-02f) {  /* MAV */
            return 1.26603812e-01f;
        } else {
            if (x[4] < 2.37086248e+00f) {  /* SNR */
                if (x[1] < -3.75030935e-02f) {  /* WL */
                    return 7.15562627e-02f;
                } else {
                    return -3.65466923e-02f;
                }
            } else {
                return -5.09979911e-02f;
            }
        }
    } else {
        if (x[2] < 1.88564217e+00f) {  /* WAMP */
            if (x[1] < -3.58789302e-02f) {  /* WL */
                if (x[3] < -3.62439528e-02f) {  /* VAR */
                    return -4.84745804e-04f;
                } else {
                    return -2.45892052e-02f;
                }
            } else {
                if (x[3] < -3.62440087e-02f) {  /* VAR */
                    return -6.31829128e-02f;
                } else {
                    return 1.62953343e-02f;
                }
            }
        } else {
            if (x[1] < -3.53919528e-02f) {  /* WL */
                if (x[10] < -4.41158749e-02f) {  /* R_Mid */
                    return -3.44318859e-02f;
                } else {
                    return -1.65442631e-01f;
                }
            } else {
                if (x[5] < -3.01347673e-01f) {  /* LOG */
                    return 6.62617609e-02f;
                } else {
                    return -2.09344495e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_127(const float *x) {
    if (x[5] < -9.83747840e-02f) {  /* LOG */
        if (x[3] < -3.62431556e-02f) {  /* VAR */
            if (x[0] < -4.01794463e-02f) {  /* MAV */
                if (x[4] < -1.80371988e+00f) {  /* SNR */
                    return -1.15563177e-01f;
                } else {
                    return -2.68593570e-03f;
                }
            } else {
                if (x[7] < -3.51395272e-02f) {  /* TFR_MeanEnergy */
                    return -6.81885257e-02f;
                } else {
                    return -1.98426750e-02f;
                }
            }
        } else {
            if (x[0] < -1.66312024e-01f) {  /* MAV */
                return 1.83741357e-02f;
            } else {
                return 9.12626758e-02f;
            }
        }
    } else {
        if (x[8] < -3.62458527e-02f) {  /* E_Total */
            if (x[1] < -3.73371392e-02f) {  /* WL */
                if (x[1] < -3.74724865e-02f) {  /* WL */
                    return -4.88794111e-02f;
                } else {
                    return 7.93423057e-02f;
                }
            } else {
                if (x[12] < -9.34169814e-02f) {  /* SpectralEntropy */
                    return -5.10204919e-02f;
                } else {
                    return 2.92863157e-02f;
                }
            }
        } else {
            if (x[12] < -9.76554304e-02f) {  /* SpectralEntropy */
                if (x[10] < -5.07157966e-02f) {  /* R_Mid */
                    return -4.60367231e-03f;
                } else {
                    return -7.13805631e-02f;
                }
            } else {
                if (x[1] < -3.74724865e-02f) {  /* WL */
                    return 1.32081851e-01f;
                } else {
                    return 3.08863982e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_128(const float *x) {
    if (x[14] < -4.79597263e-02f) {  /* MeanFreq */
        if (x[10] < -4.73016351e-02f) {  /* R_Mid */
            if (x[13] < -5.67353427e-01f) {  /* SpectralKurtosis */
                if (x[10] < -4.99250144e-02f) {  /* R_Mid */
                    return -2.29202453e-02f;
                } else {
                    return 4.11241204e-02f;
                }
            } else {
                if (x[10] < -4.75467108e-02f) {  /* R_Mid */
                    return 1.57878327e-03f;
                } else {
                    return -7.76152089e-02f;
                }
            }
        } else {
            if (x[5] < 7.02390075e-01f) {  /* LOG */
                if (x[4] < -4.08575267e-01f) {  /* SNR */
                    return 5.86018339e-02f;
                } else {
                    return 1.42570259e-02f;
                }
            } else {
                if (x[14] < -4.82949279e-02f) {  /* MeanFreq */
                    return -4.02503228e-03f;
                } else {
                    return -7.87348598e-02f;
                }
            }
        }
    } else {
        if (x[2] < 8.71930242e-01f) {  /* WAMP */
            if (x[10] < -4.98727895e-02f) {  /* R_Mid */
                if (x[0] < -4.87293415e-02f) {  /* MAV */
                    return 3.68593596e-02f;
                } else {
                    return -1.56968564e-01f;
                }
            } else {
                if (x[3] < -3.62438560e-02f) {  /* VAR */
                    return -3.08501460e-02f;
                } else {
                    return 2.67776102e-02f;
                }
            }
        } else {
            if (x[2] < 1.40546274e+00f) {  /* WAMP */
                if (x[14] < -4.52418663e-02f) {  /* MeanFreq */
                    return 2.27922183e-02f;
                } else {
                    return -9.54701379e-03f;
                }
            } else {
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return 1.07724098e-02f;
                } else {
                    return -2.53388230e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_129(const float *x) {
    if (x[1] < -3.73099335e-02f) {  /* WL */
        if (x[2] < -8.62050474e-01f) {  /* WAMP */
            if (x[5] < -2.83600867e-01f) {  /* LOG */
                if (x[1] < -3.73480208e-02f) {  /* WL */
                    return -6.38035918e-03f;
                } else {
                    return -8.95105451e-02f;
                }
            } else {
                if (x[1] < -3.73779461e-02f) {  /* WL */
                    return -7.03847781e-03f;
                } else {
                    return 2.32177787e-02f;
                }
            }
        } else {
            if (x[6] < -3.62458006e-02f) {  /* SSI */
                if (x[13] < 1.06494689e+00f) {  /* SpectralKurtosis */
                    return 8.73981800e-04f;
                } else {
                    return 9.45755169e-02f;
                }
            } else {
                if (x[1] < -3.73446234e-02f) {  /* WL */
                    return -7.53423944e-02f;
                } else {
                    return -2.16869302e-02f;
                }
            }
        }
    } else {
        if (x[13] < 5.77103317e-01f) {  /* SpectralKurtosis */
            if (x[12] < -9.72070992e-02f) {  /* SpectralEntropy */
                if (x[10] < -5.07961437e-02f) {  /* R_Mid */
                    return 4.91724089e-02f;
                } else {
                    return -6.85519874e-02f;
                }
            } else {
                if (x[14] < -4.99593318e-02f) {  /* MeanFreq */
                    return 6.22061547e-03f;
                } else {
                    return -1.49539940e-03f;
                }
            }
        } else {
            if (x[12] < -9.47066844e-02f) {  /* SpectralEntropy */
                if (x[10] < -5.06777093e-02f) {  /* R_Mid */
                    return 1.16993964e-01f;
                } else {
                    return -1.85745992e-02f;
                }
            } else {
                if (x[0] < -4.06270064e-02f) {  /* MAV */
                    return 1.47243869e-03f;
                } else {
                    return -1.11958101e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_130(const float *x) {
    if (x[13] < -6.26930475e-01f) {  /* SpectralKurtosis */
        if (x[10] < -4.69246693e-02f) {  /* R_Mid */
            if (x[3] < -3.62435095e-02f) {  /* VAR */
                if (x[3] < -3.62437777e-02f) {  /* VAR */
                    return 4.46231337e-03f;
                } else {
                    return 9.78210643e-02f;
                }
            } else {
                if (x[1] < -3.42500024e-02f) {  /* WL */
                    return -1.92687213e-01f;
                } else {
                    return 4.64191101e-02f;
                }
            }
        } else {
            if (x[2] < -5.95284164e-01f) {  /* WAMP */
                if (x[4] < -2.40263295e+00f) {  /* SNR */
                    return -6.42861649e-02f;
                } else {
                    return 5.08765765e-02f;
                }
            } else {
                if (x[2] < 2.36582136e+00f) {  /* WAMP */
                    return 7.52132609e-02f;
                } else {
                    return -1.29550723e-02f;
                }
            }
        }
    } else {
        if (x[3] < -3.62434052e-02f) {  /* VAR */
            if (x[1] < -3.58789302e-02f) {  /* WL */
                if (x[3] < -3.62439528e-02f) {  /* VAR */
                    return 2.40275855e-04f;
                } else {
                    return -2.91958507e-02f;
                }
            } else {
                if (x[7] < -3.51247117e-02f) {  /* TFR_MeanEnergy */
                    return 1.01805376e-02f;
                } else {
                    return 8.72103870e-02f;
                }
            }
        } else {
            if (x[10] < -4.22207415e-02f) {  /* R_Mid */
                if (x[2] < 1.85896552e+00f) {  /* WAMP */
                    return -3.21465991e-02f;
                } else {
                    return 7.94782043e-02f;
                }
            } else {
                if (x[6] < -3.62357050e-02f) {  /* SSI */
                    return -1.05157577e-01f;
                } else {
                    return 7.95704406e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_131(const float *x) {
    if (x[10] < -5.07524684e-02f) {  /* R_Mid */
        if (x[1] < -3.72419208e-02f) {  /* WL */
            if (x[2] < -7.82020569e-01f) {  /* WAMP */
                if (x[14] < -5.04878275e-02f) {  /* MeanFreq */
                    return 1.16521288e-02f;
                } else {
                    return -1.15863038e-02f;
                }
            } else {
                if (x[10] < -5.08003086e-02f) {  /* R_Mid */
                    return 1.32411914e-02f;
                } else {
                    return -8.28463063e-02f;
                }
            }
        } else {
            if (x[7] < -3.51409428e-02f) {  /* TFR_MeanEnergy */
                return 1.66459799e-01f;
            } else {
                if (x[6] < -3.62448730e-02f) {  /* SSI */
                    return -2.31914707e-02f;
                } else {
                    return 6.03887737e-02f;
                }
            }
        }
    } else {
        if (x[14] < -5.03358580e-02f) {  /* MeanFreq */
            if (x[1] < -3.73269394e-02f) {  /* WL */
                if (x[10] < -5.06273359e-02f) {  /* R_Mid */
                    return 8.90409574e-03f;
                } else {
                    return -2.69130152e-02f;
                }
            } else {
                if (x[12] < -9.49216411e-02f) {  /* SpectralEntropy */
                    return 2.17634682e-02f;
                } else {
                    return 7.93462768e-02f;
                }
            }
        } else {
            if (x[2] < -7.55343974e-01f) {  /* WAMP */
                if (x[7] < -3.51379551e-02f) {  /* TFR_MeanEnergy */
                    return -2.35601887e-03f;
                } else {
                    return -4.32028547e-02f;
                }
            } else {
                if (x[10] < -5.07409908e-02f) {  /* R_Mid */
                    return -9.75478515e-02f;
                } else {
                    return 3.88703472e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_132(const float *x) {
    if (x[1] < -3.58789302e-02f) {  /* WL */
        if (x[14] < -4.59191911e-02f) {  /* MeanFreq */
            if (x[14] < -4.64126877e-02f) {  /* MeanFreq */
                if (x[6] < -3.62415910e-02f) {  /* SSI */
                    return -5.46172078e-06f;
                } else {
                    return -1.50847901e-02f;
                }
            } else {
                if (x[5] < 1.04134607e+00f) {  /* LOG */
                    return 4.77826521e-02f;
                } else {
                    return -7.06402659e-02f;
                }
            }
        } else {
            if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                if (x[9] < -3.62691209e-02f) {  /* E_Low */
                    return -1.75485946e-02f;
                } else {
                    return -6.85678571e-02f;
                }
            } else {
                if (x[1] < -3.59829925e-02f) {  /* WL */
                    return 8.49876925e-02f;
                } else {
                    return 1.08769452e-02f;
                }
            }
        }
    } else {
        if (x[14] < -4.60544825e-02f) {  /* MeanFreq */
            if (x[0] < -1.61885489e-02f) {  /* MAV */
                if (x[10] < -4.78219502e-02f) {  /* R_Mid */
                    return 1.22697584e-01f;
                } else {
                    return 3.80614810e-02f;
                }
            } else {
                if (x[13] < -6.08117580e-01f) {  /* SpectralKurtosis */
                    return 5.67507856e-02f;
                } else {
                    return -9.05294120e-02f;
                }
            }
        } else {
            if (x[7] < -3.51247117e-02f) {  /* TFR_MeanEnergy */
                if (x[5] < 1.11175334e+00f) {  /* LOG */
                    return 7.70531176e-03f;
                } else {
                    return -6.71039149e-02f;
                }
            } else {
                if (x[13] < -6.15700543e-01f) {  /* SpectralKurtosis */
                    return 2.77342238e-02f;
                } else {
                    return 1.29246950e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_133(const float *x) {
    if (x[7] < -3.51393521e-02f) {  /* TFR_MeanEnergy */
        if (x[1] < -3.75234969e-02f) {  /* WL */
            if (x[0] < -4.01329771e-02f) {  /* MAV */
                if (x[0] < -4.21993881e-02f) {  /* MAV */
                    return 2.79925317e-02f;
                } else {
                    return 1.14491701e-01f;
                }
            } else {
                return -1.68291088e-02f;
            }
        } else {
            if (x[11] < 7.94922352e-01f) {  /* TopFreq2 */
                if (x[2] < 1.77893567e+00f) {  /* WAMP */
                    return -4.23376542e-03f;
                } else {
                    return 3.45873982e-02f;
                }
            } else {
                return -9.54046994e-02f;
            }
        }
    } else {
        if (x[6] < -3.62449065e-02f) {  /* SSI */
            if (x[1] < -3.74201164e-02f) {  /* WL */
                if (x[13] < 2.21051526e+00f) {  /* SpectralKurtosis */
                    return -8.14157575e-02f;
                } else {
                    return 1.12807294e-02f;
                }
            } else {
                if (x[12] < -9.58702788e-02f) {  /* SpectralEntropy */
                    return 1.09540783e-01f;
                } else {
                    return 1.38699589e-02f;
                }
            }
        } else {
            if (x[12] < -9.94796827e-02f) {  /* SpectralEntropy */
                return -8.58636945e-02f;
            } else {
                if (x[13] < 3.64703685e-01f) {  /* SpectralKurtosis */
                    return 1.79321258e-04f;
                } else {
                    return 1.23806400e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_134(const float *x) {
    if (x[14] < -4.24311534e-02f) {  /* MeanFreq */
        if (x[10] < -4.26135026e-02f) {  /* R_Mid */
            if (x[7] < -3.51380780e-02f) {  /* TFR_MeanEnergy */
                if (x[10] < -4.32782248e-02f) {  /* R_Mid */
                    return 2.50103115e-03f;
                } else {
                    return -7.87083060e-02f;
                }
            } else {
                if (x[7] < -3.51377316e-02f) {  /* TFR_MeanEnergy */
                    return -3.10977641e-02f;
                } else {
                    return -3.14709847e-03f;
                }
            }
        } else {
            if (x[3] < -3.62438560e-02f) {  /* VAR */
                if (x[12] < -4.22793366e-02f) {  /* SpectralEntropy */
                    return 7.91872293e-02f;
                } else {
                    return 7.59437261e-03f;
                }
            } else {
                if (x[4] < -1.13760650e+00f) {  /* SNR */
                    return 1.04025580e-01f;
                } else {
                    return -2.57964563e-02f;
                }
            }
        }
    } else {
        if (x[3] < -3.62437777e-02f) {  /* VAR */
            if (x[5] < -2.03995481e-02f) {  /* LOG */
                if (x[10] < -3.31971049e-02f) {  /* R_Mid */
                    return -1.96935132e-01f;
                } else {
                    return 2.51009446e-02f;
                }
            } else {
                if (x[7] < -3.51318978e-02f) {  /* TFR_MeanEnergy */
                    return 8.04946348e-02f;
                } else {
                    return -5.62825203e-02f;
                }
            }
        } else {
            if (x[3] < -3.62437516e-02f) {  /* VAR */
                if (x[2] < 1.91231871e+00f) {  /* WAMP */
                    return 8.09738487e-02f;
                } else {
                    return 1.71834771e-02f;
                }
            } else {
                if (x[0] < -2.52643283e-02f) {  /* MAV */
                    return 7.30145432e-04f;
                } else {
                    return -3.28401588e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_135(const float *x) {
    if (x[12] < -9.89609808e-02f) {  /* SpectralEntropy */
        if (x[1] < -3.73976752e-02f) {  /* WL */
            if (x[2] < -8.88727069e-01f) {  /* WAMP */
                if (x[1] < -3.74473222e-02f) {  /* WL */
                    return -3.06096412e-02f;
                } else {
                    return 3.00970562e-02f;
                }
            } else {
                if (x[0] < -4.01329771e-02f) {  /* MAV */
                    return -1.10721521e-01f;
                } else {
                    return -4.28648107e-02f;
                }
            }
        } else {
            if (x[2] < -8.08697224e-01f) {  /* WAMP */
                if (x[4] < 1.82875657e+00f) {  /* SNR */
                    return -2.48202458e-02f;
                } else {
                    return 1.07551038e-01f;
                }
            } else {
                if (x[1] < -3.73446234e-02f) {  /* WL */
                    return -8.64867866e-02f;
                } else {
                    return 6.26545120e-03f;
                }
            }
        }
    } else {
        if (x[12] < -9.82443914e-02f) {  /* SpectralEntropy */
            if (x[1] < -3.73480208e-02f) {  /* WL */
                if (x[0] < -4.49803099e-02f) {  /* MAV */
                    return 7.50405565e-02f;
                } else {
                    return 6.68604532e-03f;
                }
            } else {
                if (x[14] < -5.04630543e-02f) {  /* MeanFreq */
                    return 1.80947468e-01f;
                } else {
                    return 1.36545124e-02f;
                }
            }
        } else {
            if (x[4] < 1.30301583e+00f) {  /* SNR */
                if (x[12] < -9.74081531e-02f) {  /* SpectralEntropy */
                    return 1.72271915e-02f;
                } else {
                    return 1.51599335e-04f;
                }
            } else {
                if (x[6] < -3.62451039e-02f) {  /* SSI */
                    return -1.60285942e-02f;
                } else {
                    return -8.15721750e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_136(const float *x) {
    if (x[1] < -3.56551632e-02f) {  /* WL */
        if (x[2] < 1.13869655e+00f) {  /* WAMP */
            if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                if (x[2] < 9.51960087e-01f) {  /* WAMP */
                    return -3.05249775e-03f;
                } else {
                    return 2.48762127e-02f;
                }
            } else {
                if (x[12] < -1.58112645e-02f) {  /* SpectralEntropy */
                    return 8.40432476e-03f;
                } else {
                    return -7.29469955e-02f;
                }
            }
        } else {
            if (x[13] < -5.87418318e-01f) {  /* SpectralKurtosis */
                if (x[6] < -3.62393558e-02f) {  /* SSI */
                    return 4.16329922e-03f;
                } else {
                    return -7.20647052e-02f;
                }
            } else {
                if (x[6] < -3.62391062e-02f) {  /* SSI */
                    return -5.75260036e-02f;
                } else {
                    return 8.58045965e-02f;
                }
            }
        }
    } else {
        if (x[4] < -9.05658960e-01f) {  /* SNR */
            if (x[6] < -3.62452157e-02f) {  /* SSI */
                if (x[8] < -3.62478532e-02f) {  /* E_Total */
                    return 2.21299808e-02f;
                } else {
                    return -1.26764819e-01f;
                }
            } else {
                if (x[7] < -3.51337418e-02f) {  /* TFR_MeanEnergy */
                    return 4.51913252e-02f;
                } else {
                    return 6.54099509e-03f;
                }
            }
        } else {
            if (x[13] < -6.09025300e-01f) {  /* SpectralKurtosis */
                return 1.02396232e-04f;
            } else {
                return -1.21495582e-01f;
            }
        }
    }
}

static inline float xgb_tree_137(const float *x) {
    if (x[11] < 5.80264390e-01f) {  /* TopFreq2 */
        if (x[2] < -9.95433569e-01f) {  /* WAMP */
            if (x[6] < -3.62445228e-02f) {  /* SSI */
                if (x[7] < -3.51408869e-02f) {  /* TFR_MeanEnergy */
                    return -1.59064643e-02f;
                } else {
                    return 3.14492509e-02f;
                }
            } else {
                if (x[1] < -3.74309979e-02f) {  /* WL */
                    return -5.92143135e-03f;
                } else {
                    return -9.76335183e-02f;
                }
            }
        } else {
            if (x[5] < -2.47573555e-01f) {  /* LOG */
                if (x[2] < -9.42080319e-01f) {  /* WAMP */
                    return -2.46104281e-02f;
                } else {
                    return 7.73219764e-03f;
                }
            } else {
                if (x[5] < -1.04050420e-01f) {  /* LOG */
                    return -1.06787253e-02f;
                } else {
                    return 7.94322172e-04f;
                }
            }
        }
    } else {
        if (x[1] < -3.71562243e-02f) {  /* WL */
            if (x[1] < -3.73976752e-02f) {  /* WL */
                return 6.93565980e-02f;
            } else {
                if (x[10] < -5.04655950e-02f) {  /* R_Mid */
                    return -8.70367810e-02f;
                } else {
                    return -7.48746842e-03f;
                }
            }
        } else {
            if (x[4] < -9.80864882e-01f) {  /* SNR */
                return -5.04551120e-02f;
            } else {
                if (x[0] < -4.65951823e-02f) {  /* MAV */
                    return -2.65578832e-03f;
                } else {
                    return 1.27386913e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_138(const float *x) {
    if (x[7] < -3.51386555e-02f) {  /* TFR_MeanEnergy */
        if (x[1] < -3.73718292e-02f) {  /* WL */
            if (x[12] < -9.93368179e-02f) {  /* SpectralEntropy */
                if (x[1] < -3.74663658e-02f) {  /* WL */
                    return -3.00598587e-03f;
                } else {
                    return 8.20946172e-02f;
                }
            } else {
                if (x[7] < -3.51424664e-02f) {  /* TFR_MeanEnergy */
                    return 2.16267593e-02f;
                } else {
                    return -1.38848051e-02f;
                }
            }
        } else {
            if (x[13] < 1.20123327e+00f) {  /* SpectralKurtosis */
                if (x[7] < -3.51393335e-02f) {  /* TFR_MeanEnergy */
                    return 1.25835126e-03f;
                } else {
                    return 1.36693968e-02f;
                }
            } else {
                if (x[0] < -4.05141152e-02f) {  /* MAV */
                    return -3.23102549e-02f;
                } else {
                    return 1.34580463e-01f;
                }
            }
        }
    } else {
        if (x[0] < -3.94476987e-02f) {  /* MAV */
            if (x[0] < -3.98832969e-02f) {  /* MAV */
                return 3.05697788e-03f;
            } else {
                return -2.41596371e-01f;
            }
        } else {
            if (x[13] < -5.88590384e-01f) {  /* SpectralKurtosis */
                if (x[5] < 1.81614906e-01f) {  /* LOG */
                    return 2.89026145e-02f;
                } else {
                    return 1.40894717e-03f;
                }
            } else {
                if (x[1] < -3.60816084e-02f) {  /* WL */
                    return -6.71924232e-03f;
                } else {
                    return -7.17524886e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_139(const float *x) {
    if (x[14] < -4.79597263e-02f) {  /* MeanFreq */
        if (x[3] < -3.62440757e-02f) {  /* VAR */
            if (x[2] < 7.11870492e-01f) {  /* WAMP */
                if (x[1] < -3.65998745e-02f) {  /* WL */
                    return -4.00947261e-04f;
                } else {
                    return 7.84922913e-02f;
                }
            } else {
                if (x[0] < -3.44689116e-02f) {  /* MAV */
                    return 1.41743990e-02f;
                } else {
                    return -1.04258321e-01f;
                }
            }
        } else {
            if (x[2] < 1.19204974e+00f) {  /* WAMP */
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return -2.11261809e-02f;
                } else {
                    return 1.98488645e-02f;
                }
            } else {
                if (x[10] < -4.81044054e-02f) {  /* R_Mid */
                    return -8.23605657e-02f;
                } else {
                    return 5.73574118e-02f;
                }
            }
        }
    } else {
        if (x[6] < -3.62424925e-02f) {  /* SSI */
            if (x[6] < -3.62436883e-02f) {  /* SSI */
                if (x[2] < 1.21872640e+00f) {  /* WAMP */
                    return -1.85792129e-02f;
                } else {
                    return 6.74375286e-03f;
                }
            } else {
                if (x[1] < -3.56157161e-02f) {  /* WL */
                    return -5.64642809e-02f;
                } else {
                    return 2.03459058e-02f;
                }
            }
        } else {
            if (x[10] < -5.01111038e-02f) {  /* R_Mid */
                if (x[14] < -4.76464033e-02f) {  /* MeanFreq */
                    return -7.21399486e-03f;
                } else {
                    return -1.11010969e-01f;
                }
            } else {
                if (x[0] < -2.70186439e-02f) {  /* MAV */
                    return 6.58956617e-02f;
                } else {
                    return 6.73683081e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_140(const float *x) {
    if (x[2] < 1.59219909e+00f) {  /* WAMP */
        if (x[1] < -3.60625647e-02f) {  /* WL */
            if (x[2] < 1.13869655e+00f) {  /* WAMP */
                if (x[4] < -5.04407704e-01f) {  /* SNR */
                    return 9.84957721e-03f;
                } else {
                    return -8.67348499e-05f;
                }
            } else {
                if (x[7] < -3.51326950e-02f) {  /* TFR_MeanEnergy */
                    return -4.85275127e-02f;
                } else {
                    return 7.12203085e-02f;
                }
            }
        } else {
            if (x[13] < -5.88590384e-01f) {  /* SpectralKurtosis */
                if (x[4] < -1.01587152e+00f) {  /* SNR */
                    return 6.76817447e-03f;
                } else {
                    return 5.09568527e-02f;
                }
            } else {
                if (x[0] < -2.73851957e-02f) {  /* MAV */
                    return -1.52611302e-03f;
                } else {
                    return -1.01970337e-01f;
                }
            }
        }
    } else {
        if (x[3] < -3.62439752e-02f) {  /* VAR */
            if (x[0] < -3.86256613e-02f) {  /* MAV */
                if (x[12] < -2.36843526e-02f) {  /* SpectralEntropy */
                    return 7.71559402e-02f;
                } else {
                    return -3.34863849e-02f;
                }
            } else {
                if (x[7] < -3.51344384e-02f) {  /* TFR_MeanEnergy */
                    return -1.05331279e-01f;
                } else {
                    return -3.39867733e-02f;
                }
            }
        } else {
            if (x[14] < -4.24311534e-02f) {  /* MeanFreq */
                if (x[13] < -6.16406202e-01f) {  /* SpectralKurtosis */
                    return -1.19828712e-02f;
                } else {
                    return 4.05832268e-02f;
                }
            } else {
                if (x[2] < 1.69890559e+00f) {  /* WAMP */
                    return -8.79090652e-02f;
                } else {
                    return -1.36718927e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_141(const float *x) {
    if (x[10] < -4.93846573e-02f) {  /* R_Mid */
        if (x[10] < -4.96475846e-02f) {  /* R_Mid */
            if (x[10] < -4.97110672e-02f) {  /* R_Mid */
                if (x[1] < -3.73004153e-02f) {  /* WL */
                    return -7.61784753e-03f;
                } else {
                    return -9.46736764e-05f;
                }
            } else {
                if (x[2] < 6.58517241e-01f) {  /* WAMP */
                    return 4.08120491e-02f;
                } else {
                    return -4.97474000e-02f;
                }
            }
        } else {
            if (x[1] < -3.69780324e-02f) {  /* WL */
                if (x[12] < -7.98559561e-02f) {  /* SpectralEntropy */
                    return -3.00751030e-02f;
                } else {
                    return -1.01531170e-01f;
                }
            } else {
                if (x[1] < -3.69385816e-02f) {  /* WL */
                    return 6.13377504e-02f;
                } else {
                    return -1.89758446e-02f;
                }
            }
        }
    } else {
        if (x[1] < -3.72133590e-02f) {  /* WL */
            if (x[1] < -3.73126566e-02f) {  /* WL */
                if (x[2] < -9.95433569e-01f) {  /* WAMP */
                    return -4.12186086e-02f;
                } else {
                    return 4.84240539e-02f;
                }
            } else {
                if (x[10] < -4.91685383e-02f) {  /* R_Mid */
                    return -3.24068964e-03f;
                } else {
                    return -1.16728380e-01f;
                }
            }
        } else {
            if (x[12] < -8.31052363e-02f) {  /* SpectralEntropy */
                if (x[10] < -4.92008403e-02f) {  /* R_Mid */
                    return 9.76891667e-02f;
                } else {
                    return 2.47978270e-02f;
                }
            } else {
                if (x[1] < -3.70167978e-02f) {  /* WL */
                    return -3.14675383e-02f;
                } else {
                    return 7.20708212e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_142(const float *x) {
    if (x[10] < -5.08336388e-02f) {  /* R_Mid */
        if (x[14] < -5.04990146e-02f) {  /* MeanFreq */
            if (x[13] < 5.23936605e+00f) {  /* SpectralKurtosis */
                if (x[1] < -3.75030935e-02f) {  /* WL */
                    return -8.63074884e-02f;
                } else {
                    return 1.47259317e-03f;
                }
            } else {
                if (x[4] < 2.37086248e+00f) {  /* SNR */
                    return 1.17172837e-01f;
                } else {
                    return -2.57820878e-02f;
                }
            }
        } else {
            if (x[5] < -7.91529417e-02f) {  /* LOG */
                if (x[10] < -5.08396998e-02f) {  /* R_Mid */
                    return 4.09159884e-02f;
                } else {
                    return 1.52606845e-01f;
                }
            } else {
                if (x[1] < -3.74384820e-02f) {  /* WL */
                    return 7.17621297e-02f;
                } else {
                    return -7.02096149e-02f;
                }
            }
        }
    } else {
        if (x[10] < -5.08264527e-02f) {  /* R_Mid */
            if (x[5] < 6.30395338e-02f) {  /* LOG */
                if (x[6] < -3.62447985e-02f) {  /* SSI */
                    return -3.63607146e-02f;
                } else {
                    return -9.15614292e-02f;
                }
            } else {
                return 6.64389953e-02f;
            }
        } else {
            if (x[4] < 2.37086248e+00f) {  /* SNR */
                if (x[1] < -3.40786092e-02f) {  /* WL */
                    return 4.09761735e-04f;
                } else {
                    return 3.55008468e-02f;
                }
            } else {
                if (x[2] < -8.88727069e-01f) {  /* WAMP */
                    return 1.13069549e-01f;
                } else {
                    return -1.21092582e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_143(const float *x) {
    if (x[7] < -3.51267308e-02f) {  /* TFR_MeanEnergy */
        if (x[7] < -3.51286046e-02f) {  /* TFR_MeanEnergy */
            if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                if (x[2] < -4.08547819e-01f) {  /* WAMP */
                    return -1.41279446e-03f;
                } else {
                    return 4.92388755e-03f;
                }
            } else {
                if (x[2] < 1.77893567e+00f) {  /* WAMP */
                    return -8.72628856e-03f;
                } else {
                    return -6.36575446e-02f;
                }
            }
        } else {
            if (x[14] < -4.95339893e-02f) {  /* MeanFreq */
                if (x[1] < -3.72215211e-02f) {  /* WL */
                    return -1.49577698e-02f;
                } else {
                    return 1.60086572e-01f;
                }
            } else {
                if (x[2] < 1.08534324e+00f) {  /* WAMP */
                    return -3.90034243e-02f;
                } else {
                    return 5.48247099e-02f;
                }
            }
        }
    } else {
        if (x[0] < 3.90061527e-03f) {  /* MAV */
            if (x[12] < 4.81761731e-02f) {  /* SpectralEntropy */
                if (x[4] < -8.62652361e-01f) {  /* SNR */
                    return 1.31838217e-01f;
                } else {
                    return -8.15311521e-02f;
                }
            } else {
                if (x[13] < -6.21788561e-01f) {  /* SpectralKurtosis */
                    return -3.35454978e-02f;
                } else {
                    return -1.34975791e-01f;
                }
            }
        } else {
            if (x[4] < 2.24323034e-01f) {  /* SNR */
                if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                    return -3.45842540e-02f;
                } else {
                    return 7.01918602e-02f;
                }
            } else {
                if (x[12] < -9.05127004e-02f) {  /* SpectralEntropy */
                    return -1.66438967e-02f;
                } else {
                    return 1.75523624e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_144(const float *x) {
    if (x[5] < -1.86761066e-01f) {  /* LOG */
        if (x[10] < -5.07096276e-02f) {  /* R_Mid */
            if (x[13] < 6.97020054e-01f) {  /* SpectralKurtosis */
                if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                    return 9.25835893e-02f;
                } else {
                    return 1.49454717e-02f;
                }
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 2.92509031e-02f;
                } else {
                    return -2.66788490e-02f;
                }
            }
        } else {
            if (x[4] < 1.05037916e+00f) {  /* SNR */
                if (x[2] < -9.42080319e-01f) {  /* WAMP */
                    return -4.56322283e-02f;
                } else {
                    return -6.67439168e-03f;
                }
            } else {
                if (x[10] < -5.05263433e-02f) {  /* R_Mid */
                    return 9.15313214e-02f;
                } else {
                    return -1.29271699e-02f;
                }
            }
        }
    } else {
        if (x[5] < -1.66605398e-01f) {  /* LOG */
            if (x[2] < -9.95433569e-01f) {  /* WAMP */
                if (x[4] < 1.09602833e+00f) {  /* SNR */
                    return 2.16446370e-02f;
                } else {
                    return 1.92364439e-01f;
                }
            } else {
                if (x[10] < -5.07123880e-02f) {  /* R_Mid */
                    return -3.98769863e-02f;
                } else {
                    return 2.69146692e-02f;
                }
            }
        } else {
            if (x[5] < 1.20883715e+00f) {  /* LOG */
                if (x[13] < 4.00187463e-01f) {  /* SpectralKurtosis */
                    return -1.54392596e-03f;
                } else {
                    return 7.93171767e-03f;
                }
            } else {
                if (x[3] < -3.62440981e-02f) {  /* VAR */
                    return 8.55038241e-02f;
                } else {
                    return 4.06540837e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_145(const float *x) {
    if (x[1] < -3.74037959e-02f) {  /* WL */
        if (x[13] < 3.29180932e+00f) {  /* SpectralKurtosis */
            if (x[7] < -3.51414718e-02f) {  /* TFR_MeanEnergy */
                if (x[0] < -4.68050241e-02f) {  /* MAV */
                    return 9.62903199e-04f;
                } else {
                    return 1.02199219e-01f;
                }
            } else {
                if (x[13] < 1.20123327e+00f) {  /* SpectralKurtosis */
                    return -5.24339452e-02f;
                } else {
                    return -1.79679152e-02f;
                }
            }
        } else {
            if (x[0] < -4.53202799e-02f) {  /* MAV */
                return -8.37806463e-02f;
            } else {
                if (x[0] < -4.00798470e-02f) {  /* MAV */
                    return 7.55437016e-02f;
                } else {
                    return -4.22283029e-03f;
                }
            }
        }
    } else {
        if (x[14] < -5.04824519e-02f) {  /* MeanFreq */
            if (x[0] < -3.87929790e-02f) {  /* MAV */
                if (x[0] < -4.35207896e-02f) {  /* MAV */
                    return -3.31802084e-03f;
                } else {
                    return 9.88653675e-02f;
                }
            } else {
                return -5.56534827e-02f;
            }
        } else {
            if (x[12] < -9.86877456e-02f) {  /* SpectralEntropy */
                if (x[14] < -5.04774898e-02f) {  /* MeanFreq */
                    return 1.30151333e-02f;
                } else {
                    return -7.35175535e-02f;
                }
            } else {
                if (x[0] < 4.67124060e-02f) {  /* MAV */
                    return -2.57820677e-04f;
                } else {
                    return -4.82095368e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_146(const float *x) {
    if (x[8] < -3.62473913e-02f) {  /* E_Total */
        if (x[10] < -5.08163311e-02f) {  /* R_Mid */
            if (x[5] < -5.15805483e-01f) {  /* LOG */
                return 3.79659235e-03f;
            } else {
                return -1.07457519e-01f;
            }
        } else {
            if (x[14] < -5.04878275e-02f) {  /* MeanFreq */
                if (x[1] < -3.74609269e-02f) {  /* WL */
                    return 8.30838084e-03f;
                } else {
                    return 1.75729677e-01f;
                }
            } else {
                if (x[5] < -2.52562463e-01f) {  /* LOG */
                    return -6.37841970e-03f;
                } else {
                    return -3.85485403e-02f;
                }
            }
        }
    } else {
        if (x[8] < -3.62472534e-02f) {  /* E_Total */
            if (x[2] < 1.59219909e+00f) {  /* WAMP */
                if (x[6] < -3.62457298e-02f) {  /* SSI */
                    return -4.61455062e-03f;
                } else {
                    return 7.43505284e-02f;
                }
            } else {
                return -9.89474282e-02f;
            }
        } else {
            if (x[2] < -9.95433569e-01f) {  /* WAMP */
                if (x[6] < -3.62453498e-02f) {  /* SSI */
                    return 8.40220675e-02f;
                } else {
                    return 1.48347188e-02f;
                }
            } else {
                if (x[1] < -3.74663658e-02f) {  /* WL */
                    return -4.36150432e-02f;
                } else {
                    return 1.41938624e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_147(const float *x) {
    if (x[5] < -4.81943178e+00f) {  /* LOG */
        if (x[2] < 1.82782058e-02f) {  /* WAMP */
            if (x[2] < -9.95433569e-01f) {  /* WAMP */
                if (x[10] < -4.80286777e-02f) {  /* R_Mid */
                    return 5.02346270e-02f;
                } else {
                    return -3.07457130e-02f;
                }
            } else {
                if (x[2] < -7.28667319e-01f) {  /* WAMP */
                    return 1.06650651e-01f;
                } else {
                    return 2.77991500e-02f;
                }
            }
        } else {
            return -4.17323112e-02f;
        }
    } else {
        if (x[5] < -3.40174288e-01f) {  /* LOG */
            if (x[10] < -4.79534864e-02f) {  /* R_Mid */
                if (x[4] < -5.97049057e-01f) {  /* SNR */
                    return 2.60361340e-02f;
                } else {
                    return -8.20564758e-03f;
                }
            } else {
                if (x[10] < -4.74665686e-02f) {  /* R_Mid */
                    return -8.15854967e-02f;
                } else {
                    return -1.66007113e-02f;
                }
            }
        } else {
            if (x[9] < -3.62753011e-02f) {  /* E_Low */
                if (x[4] < 1.38299870e+00f) {  /* SNR */
                    return 1.32438960e-02f;
                } else {
                    return 9.03695300e-02f;
                }
            } else {
                if (x[7] < -3.51408385e-02f) {  /* TFR_MeanEnergy */
                    return -4.52318378e-02f;
                } else {
                    return -1.75012462e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_148(const float *x) {
    if (x[13] < -4.14937586e-01f) {  /* SpectralKurtosis */
        if (x[14] < -4.94047999e-02f) {  /* MeanFreq */
            if (x[6] < -3.62450816e-02f) {  /* SSI */
                if (x[14] < -4.96097989e-02f) {  /* MeanFreq */
                    return 2.97710188e-02f;
                } else {
                    return -2.30007805e-02f;
                }
            } else {
                if (x[4] < 2.33398885e-01f) {  /* SNR */
                    return -4.10845578e-02f;
                } else {
                    return 7.12597072e-02f;
                }
            }
        } else {
            if (x[13] < -4.75236863e-01f) {  /* SpectralKurtosis */
                if (x[14] < -4.93557937e-02f) {  /* MeanFreq */
                    return 4.04674374e-02f;
                } else {
                    return -1.72270613e-03f;
                }
            } else {
                if (x[12] < -6.98539615e-02f) {  /* SpectralEntropy */
                    return -1.46357585e-02f;
                } else {
                    return 7.85576031e-02f;
                }
            }
        }
    } else {
        if (x[12] < -7.40021467e-02f) {  /* SpectralEntropy */
            if (x[7] < -3.51297967e-02f) {  /* TFR_MeanEnergy */
                if (x[1] < -3.70521657e-02f) {  /* WL */
                    return 2.83022574e-03f;
                } else {
                    return -1.41420858e-02f;
                }
            } else {
                if (x[6] < -3.62258814e-02f) {  /* SSI */
                    return 7.10690022e-02f;
                } else {
                    return -5.42170182e-02f;
                }
            }
        } else {
            if (x[1] < -3.70848104e-02f) {  /* WL */
                if (x[3] < -3.62440795e-02f) {  /* VAR */
                    return 7.85536394e-02f;
                } else {
                    return -1.22596763e-01f;
                }
            } else {
                if (x[14] < -4.97719795e-02f) {  /* MeanFreq */
                    return 1.35923354e-02f;
                } else {
                    return 1.41202152e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_149(const float *x) {
    if (x[2] < -8.08697224e-01f) {  /* WAMP */
        if (x[14] < -5.01639098e-02f) {  /* MeanFreq */
            if (x[14] < -5.01721092e-02f) {  /* MeanFreq */
                if (x[13] < -2.38630101e-01f) {  /* SpectralKurtosis */
                    return 9.38953459e-02f;
                } else {
                    return 5.69584221e-03f;
                }
            } else {
                if (x[13] < -2.28910401e-01f) {  /* SpectralKurtosis */
                    return 1.70939434e-02f;
                } else {
                    return 1.60477802e-01f;
                }
            }
        } else {
            if (x[1] < -3.73643450e-02f) {  /* WL */
                if (x[3] < -3.62441055e-02f) {  /* VAR */
                    return -1.23252850e-02f;
                } else {
                    return 1.17896758e-01f;
                }
            } else {
                if (x[1] < -3.72759290e-02f) {  /* WL */
                    return -6.24654181e-02f;
                } else {
                    return -6.57853577e-03f;
                }
            }
        }
    } else {
        if (x[1] < -3.72915752e-02f) {  /* WL */
            if (x[13] < 6.03853703e-01f) {  /* SpectralKurtosis */
                if (x[1] < -3.73548269e-02f) {  /* WL */
                    return 5.90216592e-02f;
                } else {
                    return -6.66426122e-02f;
                }
            } else {
                if (x[1] < -3.73446234e-02f) {  /* WL */
                    return -3.52799520e-02f;
                } else {
                    return 4.76262048e-02f;
                }
            }
        } else {
            if (x[13] < 2.75487185e-01f) {  /* SpectralKurtosis */
                if (x[3] < -3.62441204e-02f) {  /* VAR */
                    return -2.62735225e-02f;
                } else {
                    return -9.86214611e-04f;
                }
            } else {
                if (x[1] < -3.72133590e-02f) {  /* WL */
                    return 2.23646089e-02f;
                } else {
                    return 1.55266598e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_150(const float *x) {
    if (x[1] < -3.74037959e-02f) {  /* WL */
        if (x[2] < -9.15403724e-01f) {  /* WAMP */
            if (x[12] < -9.72737670e-02f) {  /* SpectralEntropy */
                if (x[1] < -3.74309979e-02f) {  /* WL */
                    return -7.86065962e-03f;
                } else {
                    return 3.35883163e-02f;
                }
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return -7.00239325e-04f;
                } else {
                    return -4.32939790e-02f;
                }
            }
        } else {
            if (x[5] < -2.67533481e-01f) {  /* LOG */
                if (x[13] < 1.66717982e+00f) {  /* SpectralKurtosis */
                    return 3.32610845e-03f;
                } else {
                    return 1.46201655e-01f;
                }
            } else {
                if (x[5] < -1.07746109e-01f) {  /* LOG */
                    return -1.01042271e-01f;
                } else {
                    return -3.10770217e-02f;
                }
            }
        }
    } else {
        if (x[13] < 8.70487034e-01f) {  /* SpectralKurtosis */
            if (x[1] < -3.73915508e-02f) {  /* WL */
                if (x[13] < 5.55300772e-01f) {  /* SpectralKurtosis */
                    return -9.28437188e-02f;
                } else {
                    return 2.05167122e-02f;
                }
            } else {
                if (x[13] < 6.49052083e-01f) {  /* SpectralKurtosis */
                    return -1.06316249e-04f;
                } else {
                    return -2.22175401e-02f;
                }
            }
        } else {
            if (x[1] < -3.73507440e-02f) {  /* WL */
                if (x[0] < -3.85114439e-02f) {  /* MAV */
                    return 2.57616919e-02f;
                } else {
                    return -3.57665122e-02f;
                }
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 1.48451120e-01f;
                } else {
                    return 4.75668870e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_151(const float *x) {
    if (x[12] < -8.11554939e-02f) {  /* SpectralEntropy */
        if (x[14] < -4.96958196e-02f) {  /* MeanFreq */
            if (x[10] < -4.93846573e-02f) {  /* R_Mid */
                if (x[10] < -4.94994484e-02f) {  /* R_Mid */
                    return -3.98753583e-03f;
                } else {
                    return -6.29144311e-02f;
                }
            } else {
                if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                    return -2.65616574e-03f;
                } else {
                    return 6.42689243e-02f;
                }
            }
        } else {
            if (x[9] < -3.62564698e-02f) {  /* E_Low */
                if (x[3] < -3.62441093e-02f) {  /* VAR */
                    return 2.58680973e-02f;
                } else {
                    return -1.20949820e-01f;
                }
            } else {
                if (x[0] < -3.68221924e-02f) {  /* MAV */
                    return 7.20972940e-02f;
                } else {
                    return -5.54829538e-02f;
                }
            }
        }
    } else {
        if (x[13] < -5.03881514e-01f) {  /* SpectralKurtosis */
            if (x[10] < -5.03597595e-02f) {  /* R_Mid */
                if (x[0] < -5.71862347e-02f) {  /* MAV */
                    return 9.72920284e-02f;
                } else {
                    return -4.48154248e-02f;
                }
            } else {
                if (x[14] < -4.93331552e-02f) {  /* MeanFreq */
                    return 3.59861255e-02f;
                } else {
                    return -1.44403463e-03f;
                }
            }
        } else {
            if (x[10] < -5.05823828e-02f) {  /* R_Mid */
                if (x[14] < -4.97281924e-02f) {  /* MeanFreq */
                    return 1.80099700e-02f;
                } else {
                    return 1.09031104e-01f;
                }
            } else {
                if (x[0] < -7.51226470e-02f) {  /* MAV */
                    return 7.25379139e-02f;
                } else {
                    return 9.11090709e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_152(const float *x) {
    if (x[2] < 1.88564217e+00f) {  /* WAMP */
        if (x[10] < -2.43523027e-04f) {  /* R_Mid */
            if (x[10] < -1.40441740e-02f) {  /* R_Mid */
                if (x[12] < 8.18656161e-02f) {  /* SpectralEntropy */
                    return 1.33952114e-03f;
                } else {
                    return -2.02779919e-02f;
                }
            } else {
                if (x[13] < -6.23944223e-01f) {  /* SpectralKurtosis */
                    return -2.94231772e-02f;
                } else {
                    return -1.28974795e-01f;
                }
            }
        } else {
            if (x[2] < -3.81871194e-01f) {  /* WAMP */
                if (x[1] < -3.62543650e-02f) {  /* WL */
                    return 4.21729051e-02f;
                } else {
                    return -9.17875543e-02f;
                }
            } else {
                if (x[2] < 1.27207959e+00f) {  /* WAMP */
                    return 1.17719665e-01f;
                } else {
                    return -1.08042555e-02f;
                }
            }
        }
    } else {
        if (x[1] < -3.53919528e-02f) {  /* WL */
            if (x[13] < -6.18349969e-01f) {  /* SpectralKurtosis */
                return 3.02814972e-02f;
            } else {
                if (x[7] < -3.51316892e-02f) {  /* TFR_MeanEnergy */
                    return -1.58303186e-01f;
                } else {
                    return -2.20417324e-02f;
                }
            }
        } else {
            if (x[7] < -3.51309516e-02f) {  /* TFR_MeanEnergy */
                if (x[2] < 1.91231871e+00f) {  /* WAMP */
                    return -7.29432479e-02f;
                } else {
                    return 2.84333471e-02f;
                }
            } else {
                if (x[0] < -1.42323477e-02f) {  /* MAV */
                    return -1.98763877e-01f;
                } else {
                    return -1.22844912e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_153(const float *x) {
    if (x[6] < -3.62459570e-02f) {  /* SSI */
        if (x[8] < -3.62478532e-02f) {  /* E_Total */
            if (x[13] < -5.08659482e-01f) {  /* SpectralKurtosis */
                if (x[4] < -7.40554512e-01f) {  /* SNR */
                    return 1.15261069e-02f;
                } else {
                    return -2.12064143e-02f;
                }
            } else {
                if (x[5] < -5.65673232e-01f) {  /* LOG */
                    return 2.58980989e-02f;
                } else {
                    return 3.71591235e-03f;
                }
            }
        } else {
            if (x[8] < -3.62478010e-02f) {  /* E_Total */
                if (x[7] < -3.51408869e-02f) {  /* TFR_MeanEnergy */
                    return 1.44471973e-01f;
                } else {
                    return 2.92102545e-02f;
                }
            } else {
                return -3.88464704e-02f;
            }
        }
    } else {
        if (x[8] < -3.62473428e-02f) {  /* E_Total */
            if (x[12] < 1.95077602e-02f) {  /* SpectralEntropy */
                if (x[1] < -3.64121534e-02f) {  /* WL */
                    return -2.41971724e-02f;
                } else {
                    return 6.31322861e-02f;
                }
            } else {
                if (x[12] < 8.89217183e-02f) {  /* SpectralEntropy */
                    return -1.49428993e-01f;
                } else {
                    return -7.98899401e-03f;
                }
            }
        } else {
            if (x[0] < -4.35207896e-02f) {  /* MAV */
                if (x[13] < -6.24451697e-01f) {  /* SpectralKurtosis */
                    return 7.05179945e-03f;
                } else {
                    return 9.28131267e-02f;
                }
            } else {
                if (x[13] < -5.88590384e-01f) {  /* SpectralKurtosis */
                    return 7.35058635e-03f;
                } else {
                    return -2.59727123e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_154(const float *x) {
    if (x[2] < -9.95433569e-01f) {  /* WAMP */
        if (x[0] < -3.72657441e-02f) {  /* MAV */
            if (x[0] < -3.82060073e-02f) {  /* MAV */
                if (x[1] < -3.73779461e-02f) {  /* WL */
                    return 2.23112060e-03f;
                } else {
                    return 7.40131363e-02f;
                }
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 9.26061010e-04f;
                } else {
                    return 1.15596831e-01f;
                }
            }
        } else {
            if (x[7] < -3.51384915e-02f) {  /* TFR_MeanEnergy */
                return -9.87573043e-02f;
            } else {
                if (x[1] < -3.74473222e-02f) {  /* WL */
                    return 2.77510174e-02f;
                } else {
                    return -4.83953580e-02f;
                }
            }
        }
    } else {
        if (x[10] < -5.06777093e-02f) {  /* R_Mid */
            if (x[13] < 2.61404872e-01f) {  /* SpectralKurtosis */
                if (x[14] < -5.02071157e-02f) {  /* MeanFreq */
                    return -5.76184951e-02f;
                } else {
                    return 1.11730164e-02f;
                }
            } else {
                if (x[0] < -5.26270978e-02f) {  /* MAV */
                    return 7.98897892e-02f;
                } else {
                    return -2.07614875e-03f;
                }
            }
        } else {
            if (x[13] < 8.14111456e-02f) {  /* SpectralKurtosis */
                if (x[11] < 5.80264390e-01f) {  /* TopFreq2 */
                    return 5.26924559e-04f;
                } else {
                    return 5.87470606e-02f;
                }
            } else {
                if (x[10] < -5.04331663e-02f) {  /* R_Mid */
                    return 9.00938082e-03f;
                } else {
                    return 6.66314960e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_155(const float *x) {
    if (x[6] < -3.62451226e-02f) {  /* SSI */
        if (x[6] < -3.62451337e-02f) {  /* SSI */
            if (x[2] < 1.19204974e+00f) {  /* WAMP */
                if (x[1] < -3.59068140e-02f) {  /* WL */
                    return -4.93871327e-03f;
                } else {
                    return -1.28253207e-01f;
                }
            } else {
                if (x[13] < -6.04469478e-01f) {  /* SpectralKurtosis */
                    return -3.29120830e-03f;
                } else {
                    return 8.37282240e-02f;
                }
            }
        } else {
            if (x[10] < -4.91023399e-02f) {  /* R_Mid */
                if (x[7] < -3.51396985e-02f) {  /* TFR_MeanEnergy */
                    return 2.74038147e-02f;
                } else {
                    return -8.83200318e-02f;
                }
            } else {
                return 2.02185214e-02f;
            }
        }
    } else {
        if (x[6] < -3.62408496e-02f) {  /* SSI */
            if (x[1] < -3.75030935e-02f) {  /* WL */
                if (x[7] < -3.51389386e-02f) {  /* TFR_MeanEnergy */
                    return -7.63572380e-02f;
                } else {
                    return 2.39975508e-02f;
                }
            } else {
                if (x[13] < 1.03015113e+00f) {  /* SpectralKurtosis */
                    return 3.23207909e-03f;
                } else {
                    return 1.80341788e-02f;
                }
            }
        } else {
            if (x[14] < -5.01246415e-02f) {  /* MeanFreq */
                if (x[2] < -8.35373819e-01f) {  /* WAMP */
                    return -3.45478095e-02f;
                } else {
                    return 1.06378160e-01f;
                }
            } else {
                if (x[10] < -4.98145372e-02f) {  /* R_Mid */
                    return -5.52154481e-02f;
                } else {
                    return -5.79994312e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_156(const float *x) {
    if (x[2] < 2.07237840e+00f) {  /* WAMP */
        if (x[14] < -5.05093522e-02f) {  /* MeanFreq */
            if (x[2] < -9.68756974e-01f) {  /* WAMP */
                if (x[0] < -4.04025801e-02f) {  /* MAV */
                    return 2.82759895e-04f;
                } else {
                    return -7.70179704e-02f;
                }
            } else {
                if (x[14] < -5.05116545e-02f) {  /* MeanFreq */
                    return 8.36521611e-02f;
                } else {
                    return -3.40016447e-02f;
                }
            }
        } else {
            if (x[5] < -2.89823115e-01f) {  /* LOG */
                if (x[9] < -3.62785198e-02f) {  /* E_Low */
                    return -3.71853262e-03f;
                } else {
                    return -3.77885252e-02f;
                }
            } else {
                if (x[9] < -3.62753011e-02f) {  /* E_Low */
                    return 3.82921398e-02f;
                } else {
                    return -4.15355637e-04f;
                }
            }
        }
    } else {
        if (x[10] < -4.18464281e-02f) {  /* R_Mid */
            if (x[10] < -4.68342118e-02f) {  /* R_Mid */
                if (x[0] < -7.07823411e-03f) {  /* MAV */
                    return 7.41438791e-02f;
                } else {
                    return 9.93701047e-04f;
                }
            } else {
                if (x[6] < -3.62407565e-02f) {  /* SSI */
                    return 1.95322409e-02f;
                } else {
                    return -7.77629614e-02f;
                }
            }
        } else {
            if (x[3] < -3.62424217e-02f) {  /* VAR */
                if (x[5] < -1.11748785e-01f) {  /* LOG */
                    return 1.95874479e-02f;
                } else {
                    return 8.78492072e-02f;
                }
            } else {
                return -2.03492213e-02f;
            }
        }
    }
}

static inline float xgb_tree_157(const float *x) {
    if (x[1] < -3.36644053e-02f) {  /* WL */
        if (x[2] < 2.20576167e+00f) {  /* WAMP */
            if (x[12] < 1.93978921e-01f) {  /* SpectralEntropy */
                if (x[1] < -3.74037959e-02f) {  /* WL */
                    return -9.38478764e-03f;
                } else {
                    return 8.95283360e-04f;
                }
            } else {
                if (x[3] < -3.62430066e-02f) {  /* VAR */
                    return -6.93057105e-02f;
                } else {
                    return 3.29417847e-02f;
                }
            }
        } else {
            if (x[5] < -2.22313449e-01f) {  /* LOG */
                return 5.37770763e-02f;
            } else {
                if (x[12] < 8.18656161e-02f) {  /* SpectralEntropy */
                    return 4.33101319e-03f;
                } else {
                    return -1.64594337e-01f;
                }
            }
        }
    } else {
        if (x[2] < -5.68607569e-01f) {  /* WAMP */
            return -6.26394153e-02f;
        } else {
            if (x[2] < 2.47252798e+00f) {  /* WAMP */
                if (x[10] < -4.04326543e-02f) {  /* R_Mid */
                    return 1.04455566e-02f;
                } else {
                    return 8.91425610e-02f;
                }
            } else {
                return 1.42445760e-02f;
            }
        }
    }
}

static inline float xgb_tree_158(const float *x) {
    if (x[14] < -4.91066240e-02f) {  /* MeanFreq */
        if (x[12] < -7.52863139e-02f) {  /* SpectralEntropy */
            if (x[14] < -5.02355546e-02f) {  /* MeanFreq */
                if (x[1] < -3.72888483e-02f) {  /* WL */
                    return -9.31325543e-04f;
                } else {
                    return 1.89876687e-02f;
                }
            } else {
                if (x[1] < -3.72174382e-02f) {  /* WL */
                    return -2.78942138e-02f;
                } else {
                    return -5.33325656e-04f;
                }
            }
        } else {
            if (x[5] < 2.21134692e-01f) {  /* LOG */
                if (x[0] < -3.17889452e-02f) {  /* MAV */
                    return 1.15907257e-02f;
                } else {
                    return -1.22227989e-01f;
                }
            } else {
                if (x[4] < -3.44080597e-01f) {  /* SNR */
                    return 1.41359298e-02f;
                } else {
                    return 8.92422050e-02f;
                }
            }
        }
    } else {
        if (x[12] < -6.50291964e-02f) {  /* SpectralEntropy */
            if (x[10] < -4.77673225e-02f) {  /* R_Mid */
                if (x[10] < -4.84978519e-02f) {  /* R_Mid */
                    return -3.69383134e-02f;
                } else {
                    return -1.16660811e-01f;
                }
            } else {
                if (x[12] < -6.73046038e-02f) {  /* SpectralEntropy */
                    return -4.26537497e-03f;
                } else {
                    return 7.10705668e-02f;
                }
            }
        } else {
            if (x[2] < 8.71930242e-01f) {  /* WAMP */
                if (x[3] < -3.62440459e-02f) {  /* VAR */
                    return 9.55930352e-03f;
                } else {
                    return -2.56999563e-02f;
                }
            } else {
                if (x[2] < 1.13869655e+00f) {  /* WAMP */
                    return 2.10034214e-02f;
                } else {
                    return -1.10628211e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_159(const float *x) {
    if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
        if (x[10] < -5.06175347e-02f) {  /* R_Mid */
            if (x[4] < 9.98444915e-01f) {  /* SNR */
                if (x[7] < -3.51404697e-02f) {  /* TFR_MeanEnergy */
                    return 1.25409728e-02f;
                } else {
                    return 7.28902593e-02f;
                }
            } else {
                if (x[7] < -3.51390913e-02f) {  /* TFR_MeanEnergy */
                    return 1.24731231e-02f;
                } else {
                    return -5.54687642e-02f;
                }
            }
        } else {
            if (x[0] < -2.94383168e-02f) {  /* MAV */
                if (x[7] < -3.51442732e-02f) {  /* TFR_MeanEnergy */
                    return 1.56531502e-02f;
                } else {
                    return -2.52830293e-02f;
                }
            } else {
                if (x[13] < -2.54293323e-01f) {  /* SpectralKurtosis */
                    return 1.00603821e-02f;
                } else {
                    return 1.23433493e-01f;
                }
            }
        }
    } else {
        if (x[13] < -7.12004676e-02f) {  /* SpectralKurtosis */
            if (x[13] < -9.07716528e-02f) {  /* SpectralKurtosis */
                if (x[4] < 6.27685070e-01f) {  /* SNR */
                    return -3.37377423e-04f;
                } else {
                    return 4.61286753e-02f;
                }
            } else {
                if (x[2] < -4.88577694e-01f) {  /* WAMP */
                    return 4.87756059e-02f;
                } else {
                    return -9.62737650e-02f;
                }
            }
        } else {
            if (x[7] < -3.51428129e-02f) {  /* TFR_MeanEnergy */
                if (x[2] < -9.15403724e-01f) {  /* WAMP */
                    return -1.73978563e-02f;
                } else {
                    return 8.41663405e-02f;
                }
            } else {
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return -1.37745189e-02f;
                } else {
                    return 7.38088740e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_160(const float *x) {
    if (x[10] < -5.07192351e-02f) {  /* R_Mid */
        if (x[4] < -1.26202568e-01f) {  /* SNR */
            return 9.82093662e-02f;
        } else {
            if (x[13] < 7.74978697e-01f) {  /* SpectralKurtosis */
                if (x[7] < -3.51411514e-02f) {  /* TFR_MeanEnergy */
                    return 3.80525030e-02f;
                } else {
                    return -2.54793856e-02f;
                }
            } else {
                if (x[0] < -5.71862347e-02f) {  /* MAV */
                    return -9.27657038e-02f;
                } else {
                    return 3.33696767e-03f;
                }
            }
        }
    } else {
        if (x[14] < -5.04653454e-02f) {  /* MeanFreq */
            if (x[7] < -3.51393856e-02f) {  /* TFR_MeanEnergy */
                return 1.35277301e-01f;
            } else {
                if (x[4] < 1.41474891e+00f) {  /* SNR */
                    return 5.48360832e-02f;
                } else {
                    return -6.88452050e-02f;
                }
            }
        } else {
            if (x[12] < -9.81095955e-02f) {  /* SpectralEntropy */
                if (x[0] < -4.56642322e-02f) {  /* MAV */
                    return 7.53556192e-03f;
                } else {
                    return -9.81369764e-02f;
                }
            } else {
                if (x[14] < -5.03358580e-02f) {  /* MeanFreq */
                    return 1.38838291e-02f;
                } else {
                    return 1.65781158e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_161(const float *x) {
    if (x[5] < -4.81943178e+00f) {  /* LOG */
        if (x[2] < -2.21811429e-01f) {  /* WAMP */
            if (x[1] < -3.74935716e-02f) {  /* WL */
                return 7.21946731e-03f;
            } else {
                if (x[2] < -6.75314069e-01f) {  /* WAMP */
                    return 9.67389196e-02f;
                } else {
                    return 3.71132158e-02f;
                }
            }
        } else {
            return -3.86059880e-02f;
        }
    } else {
        if (x[1] < -3.73126566e-02f) {  /* WL */
            if (x[14] < -5.03934473e-02f) {  /* MeanFreq */
                if (x[12] < -9.57963988e-02f) {  /* SpectralEntropy */
                    return -4.24866425e-03f;
                } else {
                    return 5.04591130e-02f;
                }
            } else {
                if (x[12] < -9.65579078e-02f) {  /* SpectralEntropy */
                    return 7.22818151e-02f;
                } else {
                    return -2.53151935e-02f;
                }
            }
        } else {
            if (x[13] < 2.75487185e-01f) {  /* SpectralKurtosis */
                if (x[5] < -3.07486445e-01f) {  /* LOG */
                    return 7.87316449e-03f;
                } else {
                    return -2.84643541e-03f;
                }
            } else {
                if (x[2] < -7.28667319e-01f) {  /* WAMP */
                    return 3.37889902e-02f;
                } else {
                    return -3.46505642e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_162(const float *x) {
    if (x[13] < -6.18349969e-01f) {  /* SpectralKurtosis */
        if (x[3] < -3.62438895e-02f) {  /* VAR */
            if (x[2] < 4.45104212e-01f) {  /* WAMP */
                return -4.77172993e-02f;
            } else {
                if (x[13] < -6.22344434e-01f) {  /* SpectralKurtosis */
                    return 3.10422219e-02f;
                } else {
                    return 8.24421868e-02f;
                }
            }
        } else {
            if (x[14] < -4.46994491e-02f) {  /* MeanFreq */
                if (x[14] < -4.50671427e-02f) {  /* MeanFreq */
                    return -2.06659418e-02f;
                } else {
                    return -1.47971883e-01f;
                }
            } else {
                if (x[14] < -4.35333475e-02f) {  /* MeanFreq */
                    return 7.79813007e-02f;
                } else {
                    return 5.65755507e-03f;
                }
            }
        }
    } else {
        if (x[13] < -6.16406202e-01f) {  /* SpectralKurtosis */
            if (x[6] < -3.62452790e-02f) {  /* SSI */
                if (x[14] < -4.52418663e-02f) {  /* MeanFreq */
                    return 2.60658301e-02f;
                } else {
                    return -1.36346489e-01f;
                }
            } else {
                if (x[7] < -3.51386815e-02f) {  /* TFR_MeanEnergy */
                    return 8.30431283e-02f;
                } else {
                    return -5.07266782e-02f;
                }
            }
        } else {
            if (x[2] < 1.91231871e+00f) {  /* WAMP */
                if (x[1] < -3.53484228e-02f) {  /* WL */
                    return 4.44538950e-04f;
                } else {
                    return 6.90473542e-02f;
                }
            } else {
                if (x[0] < -2.55910102e-02f) {  /* MAV */
                    return -9.87469256e-02f;
                } else {
                    return 4.94847773e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_163(const float *x) {
    if (x[12] < -8.89153630e-02f) {  /* SpectralEntropy */
        if (x[9] < -3.62352207e-02f) {  /* E_Low */
            if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                if (x[14] < -5.02355546e-02f) {  /* MeanFreq */
                    return -4.79532784e-04f;
                } else {
                    return -2.22334582e-02f;
                }
            } else {
                if (x[5] < -1.00986588e+00f) {  /* LOG */
                    return -9.38853770e-02f;
                } else {
                    return 1.03469379e-02f;
                }
            }
        } else {
            if (x[13] < -3.84483546e-01f) {  /* SpectralKurtosis */
                if (x[0] < -2.16653459e-02f) {  /* MAV */
                    return 1.97555184e-01f;
                } else {
                    return -2.51726843e-02f;
                }
            } else {
                if (x[14] < -5.01016416e-02f) {  /* MeanFreq */
                    return -2.22368836e-02f;
                } else {
                    return -9.49926972e-02f;
                }
            }
        }
    } else {
        if (x[14] < -5.00832237e-02f) {  /* MeanFreq */
            if (x[12] < -8.59299898e-02f) {  /* SpectralEntropy */
                if (x[7] < -3.51408012e-02f) {  /* TFR_MeanEnergy */
                    return -2.16701925e-02f;
                } else {
                    return 3.48641537e-02f;
                }
            } else {
                if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                    return 1.02795474e-01f;
                } else {
                    return -4.45018634e-02f;
                }
            }
        } else {
            if (x[6] < -3.62427570e-02f) {  /* SSI */
                if (x[7] < -3.51380780e-02f) {  /* TFR_MeanEnergy */
                    return 1.64789415e-03f;
                } else {
                    return -1.44152241e-02f;
                }
            } else {
                if (x[1] < -3.71643864e-02f) {  /* WL */
                    return -8.44357237e-02f;
                } else {
                    return 1.18709812e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_164(const float *x) {
    if (x[0] < -7.07823411e-03f) {  /* MAV */
        if (x[7] < -3.51297967e-02f) {  /* TFR_MeanEnergy */
            if (x[8] < -3.62343453e-02f) {  /* E_Total */
                if (x[7] < -3.51309516e-02f) {  /* TFR_MeanEnergy */
                    return -1.25075050e-04f;
                } else {
                    return -4.60732430e-02f;
                }
            } else {
                if (x[13] < -5.19735336e-01f) {  /* SpectralKurtosis */
                    return 8.67524073e-02f;
                } else {
                    return -9.65409819e-03f;
                }
            }
        } else {
            if (x[10] < -4.55504954e-02f) {  /* R_Mid */
                if (x[10] < -5.01111038e-02f) {  /* R_Mid */
                    return -7.77597204e-02f;
                } else {
                    return 1.16717452e-02f;
                }
            } else {
                if (x[2] < 1.61887574e+00f) {  /* WAMP */
                    return -1.24093235e-01f;
                } else {
                    return -1.86941233e-02f;
                }
            }
        }
    } else {
        if (x[7] < -3.51277366e-02f) {  /* TFR_MeanEnergy */
            if (x[13] < -5.78381002e-01f) {  /* SpectralKurtosis */
                if (x[13] < -6.08117580e-01f) {  /* SpectralKurtosis */
                    return 1.09367549e-01f;
                } else {
                    return -4.04704027e-02f;
                }
            } else {
                if (x[3] < -3.62441018e-02f) {  /* VAR */
                    return 2.75829565e-02f;
                } else {
                    return 1.65427089e-01f;
                }
            }
        } else {
            if (x[12] < 1.29795447e-02f) {  /* SpectralEntropy */
                if (x[3] < -3.62440832e-02f) {  /* VAR */
                    return 2.61324346e-02f;
                } else {
                    return -3.37996334e-02f;
                }
            } else {
                if (x[13] < -6.12127721e-01f) {  /* SpectralKurtosis */
                    return 9.20191407e-04f;
                } else {
                    return 8.04897621e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_165(const float *x) {
    if (x[7] < -3.51377316e-02f) {  /* TFR_MeanEnergy */
        if (x[9] < -3.62386703e-02f) {  /* E_Low */
            if (x[1] < -3.52960527e-02f) {  /* WL */
                if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                    return -1.71950238e-03f;
                } else {
                    return 8.87295883e-03f;
                }
            } else {
                if (x[12] < 7.45404065e-02f) {  /* SpectralEntropy */
                    return -1.59616217e-01f;
                } else {
                    return -3.65264760e-03f;
                }
            }
        } else {
            if (x[1] < -3.59360613e-02f) {  /* WL */
                if (x[2] < 7.11870492e-01f) {  /* WAMP */
                    return -3.85077894e-02f;
                } else {
                    return -2.18647555e-01f;
                }
            } else {
                return 1.74402967e-02f;
            }
        }
    } else {
        if (x[5] < 1.81614906e-01f) {  /* LOG */
            if (x[6] < -3.62440981e-02f) {  /* SSI */
                if (x[5] < 2.39668749e-02f) {  /* LOG */
                    return -1.32233858e-01f;
                } else {
                    return 3.21125314e-02f;
                }
            } else {
                if (x[2] < 1.82782058e-02f) {  /* WAMP */
                    return 6.52890885e-03f;
                } else {
                    return 4.31572981e-02f;
                }
            }
        } else {
            if (x[4] < 5.32929420e-01f) {  /* SNR */
                if (x[13] < -2.38630101e-01f) {  /* SpectralKurtosis */
                    return 1.73394964e-03f;
                } else {
                    return 2.76733935e-02f;
                }
            } else {
                if (x[12] < -9.74081531e-02f) {  /* SpectralEntropy */
                    return 3.70213129e-02f;
                } else {
                    return -5.46717905e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_166(const float *x) {
    if (x[7] < -3.51377316e-02f) {  /* TFR_MeanEnergy */
        if (x[1] < -3.52960527e-02f) {  /* WL */
            if (x[5] < -1.74581444e+00f) {  /* LOG */
                if (x[8] < -3.62651497e-02f) {  /* E_Total */
                    return 4.53558983e-03f;
                } else {
                    return 7.17596412e-02f;
                }
            } else {
                if (x[14] < -3.75789218e-02f) {  /* MeanFreq */
                    return -1.43315934e-03f;
                } else {
                    return -9.19234678e-02f;
                }
            }
        } else {
            if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                if (x[13] < -6.17754340e-01f) {  /* SpectralKurtosis */
                    return -3.27666029e-02f;
                } else {
                    return 5.67635708e-02f;
                }
            } else {
                if (x[5] < -3.13272268e-01f) {  /* LOG */
                    return 6.04385436e-02f;
                } else {
                    return -2.69804567e-01f;
                }
            }
        }
    } else {
        if (x[5] < 1.43696412e-01f) {  /* LOG */
            if (x[1] < -3.73099335e-02f) {  /* WL */
                if (x[14] < -5.04387543e-02f) {  /* MeanFreq */
                    return 6.90436289e-02f;
                } else {
                    return -6.63799196e-02f;
                }
            } else {
                if (x[13] < 3.21468944e-03f) {  /* SpectralKurtosis */
                    return 2.80279219e-02f;
                } else {
                    return 1.31451383e-01f;
                }
            }
        } else {
            if (x[1] < -3.71521451e-02f) {  /* WL */
                if (x[4] < 1.06509686e+00f) {  /* SNR */
                    return -2.11611837e-02f;
                } else {
                    return 3.42340618e-02f;
                }
            } else {
                if (x[13] < -2.38630101e-01f) {  /* SpectralKurtosis */
                    return 2.44535203e-03f;
                } else {
                    return 5.87400906e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_167(const float *x) {
    if (x[10] < -5.03810905e-02f) {  /* R_Mid */
        if (x[3] < -3.62440757e-02f) {  /* VAR */
            if (x[1] < -3.71562243e-02f) {  /* WL */
                if (x[14] < -5.02071157e-02f) {  /* MeanFreq */
                    return -3.54201463e-03f;
                } else {
                    return 2.63513308e-02f;
                }
            } else {
                if (x[7] < -3.51392552e-02f) {  /* TFR_MeanEnergy */
                    return -3.78268994e-02f;
                } else {
                    return 3.44568235e-03f;
                }
            }
        } else {
            if (x[13] < -5.06195843e-01f) {  /* SpectralKurtosis */
                if (x[7] < -3.51391584e-02f) {  /* TFR_MeanEnergy */
                    return 3.36567573e-02f;
                } else {
                    return -3.35041955e-02f;
                }
            } else {
                if (x[3] < -3.62440273e-02f) {  /* VAR */
                    return 1.12891905e-01f;
                } else {
                    return 9.64020379e-03f;
                }
            }
        }
    } else {
        if (x[14] < -5.01639098e-02f) {  /* MeanFreq */
            if (x[7] < -3.51401605e-02f) {  /* TFR_MeanEnergy */
                if (x[7] < -3.51425782e-02f) {  /* TFR_MeanEnergy */
                    return 6.00954480e-02f;
                } else {
                    return -4.04470228e-02f;
                }
            } else {
                if (x[7] < -3.51399146e-02f) {  /* TFR_MeanEnergy */
                    return 1.23341799e-01f;
                } else {
                    return 3.79670598e-02f;
                }
            }
        } else {
            if (x[14] < -5.01326248e-02f) {  /* MeanFreq */
                if (x[0] < -3.88793200e-02f) {  /* MAV */
                    return -3.71406262e-04f;
                } else {
                    return -7.91325346e-02f;
                }
            } else {
                if (x[5] < 1.68236624e-02f) {  /* LOG */
                    return 5.11336513e-03f;
                } else {
                    return -4.36860323e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_168(const float *x) {
    if (x[12] < -7.95188099e-02f) {  /* SpectralEntropy */
        if (x[13] < 3.64703685e-01f) {  /* SpectralKurtosis */
            if (x[14] < -5.03634959e-02f) {  /* MeanFreq */
                if (x[7] < -3.51404399e-02f) {  /* TFR_MeanEnergy */
                    return 3.38811316e-02f;
                } else {
                    return -5.27009144e-02f;
                }
            } else {
                if (x[13] < -5.52980542e-01f) {  /* SpectralKurtosis */
                    return 1.27621397e-01f;
                } else {
                    return -3.61919217e-03f;
                }
            }
        } else {
            if (x[1] < -3.72854508e-02f) {  /* WL */
                if (x[8] < -3.62414606e-02f) {  /* E_Total */
                    return 6.25997735e-03f;
                } else {
                    return -6.83516562e-02f;
                }
            } else {
                if (x[0] < -3.19854990e-02f) {  /* MAV */
                    return 6.03093915e-02f;
                } else {
                    return -4.50890027e-02f;
                }
            }
        }
    } else {
        if (x[13] < -3.95395875e-01f) {  /* SpectralKurtosis */
            if (x[1] < -3.71521451e-02f) {  /* WL */
                if (x[1] < -3.73215005e-02f) {  /* WL */
                    return 2.68606339e-02f;
                } else {
                    return -9.45709720e-02f;
                }
            } else {
                if (x[14] < -4.91066240e-02f) {  /* MeanFreq */
                    return 1.32986652e-02f;
                } else {
                    return 1.78649556e-03f;
                }
            }
        } else {
            if (x[3] < -3.62440832e-02f) {  /* VAR */
                if (x[10] < -5.03597595e-02f) {  /* R_Mid */
                    return 5.40128858e-05f;
                } else {
                    return 5.49666397e-02f;
                }
            } else {
                if (x[1] < -3.70848104e-02f) {  /* WL */
                    return -2.04031207e-02f;
                } else {
                    return 1.37110591e-01f;
                }
            }
        }
    }
}

static inline float xgb_tree_169(const float *x) {
    if (x[13] < -6.26930475e-01f) {  /* SpectralKurtosis */
        if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
            if (x[2] < 1.59219909e+00f) {  /* WAMP */
                if (x[4] < -2.40263295e+00f) {  /* SNR */
                    return 2.07415726e-02f;
                } else {
                    return 1.05671205e-01f;
                }
            } else {
                if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                    return -7.13755712e-02f;
                } else {
                    return 5.95715344e-02f;
                }
            }
        } else {
            if (x[12] < 7.45404065e-02f) {  /* SpectralEntropy */
                return -9.43697318e-02f;
            } else {
                if (x[0] < -1.61885489e-02f) {  /* MAV */
                    return 8.55335817e-02f;
                } else {
                    return -2.01966316e-02f;
                }
            }
        }
    } else {
        if (x[4] < -1.32085180e+00f) {  /* SNR */
            if (x[10] < -4.95262817e-02f) {  /* R_Mid */
                if (x[0] < -5.21450303e-02f) {  /* MAV */
                    return -2.83180363e-02f;
                } else {
                    return -1.62691996e-01f;
                }
            } else {
                if (x[0] < -1.26246616e-01f) {  /* MAV */
                    return 8.57076719e-02f;
                } else {
                    return -2.37537306e-02f;
                }
            }
        } else {
            if (x[10] < -5.08435890e-02f) {  /* R_Mid */
                if (x[8] < -3.62456813e-02f) {  /* E_Total */
                    return 1.10871352e-01f;
                } else {
                    return -1.08921928e-02f;
                }
            } else {
                if (x[10] < -3.15503068e-02f) {  /* R_Mid */
                    return 9.37436242e-04f;
                } else {
                    return 4.81125675e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_170(const float *x) {
    if (x[12] < -9.89609808e-02f) {  /* SpectralEntropy */
        if (x[14] < -5.04904240e-02f) {  /* MeanFreq */
            if (x[13] < 1.34037542e+00f) {  /* SpectralKurtosis */
                if (x[0] < -4.22697887e-02f) {  /* MAV */
                    return 1.15695119e-01f;
                } else {
                    return 9.29683261e-03f;
                }
            } else {
                if (x[5] < -1.30979195e-01f) {  /* LOG */
                    return -5.52478991e-02f;
                } else {
                    return -2.56636087e-03f;
                }
            }
        } else {
            if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                if (x[13] < 8.09494078e-01f) {  /* SpectralKurtosis */
                    return 5.13673723e-02f;
                } else {
                    return -6.13372214e-02f;
                }
            } else {
                if (x[14] < -5.04878275e-02f) {  /* MeanFreq */
                    return -2.02539358e-02f;
                } else {
                    return -9.40532684e-02f;
                }
            }
        }
    } else {
        if (x[14] < -5.03478497e-02f) {  /* MeanFreq */
            if (x[12] < -9.52586532e-02f) {  /* SpectralEntropy */
                if (x[4] < 7.24959373e-01f) {  /* SNR */
                    return -1.00439906e-01f;
                } else {
                    return 6.25332166e-03f;
                }
            } else {
                if (x[13] < 4.17913109e-01f) {  /* SpectralKurtosis */
                    return 9.48360041e-02f;
                } else {
                    return 1.18779596e-02f;
                }
            }
        } else {
            if (x[2] < -8.88727069e-01f) {  /* WAMP */
                if (x[12] < -9.50429589e-02f) {  /* SpectralEntropy */
                    return 6.22574091e-02f;
                } else {
                    return -3.00012883e-02f;
                }
            } else {
                if (x[13] < 5.55300772e-01f) {  /* SpectralKurtosis */
                    return -5.70133561e-05f;
                } else {
                    return -9.16941538e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_171(const float *x) {
    if (x[13] < -6.20130956e-01f) {  /* SpectralKurtosis */
        if (x[4] < -1.32085180e+00f) {  /* SNR */
            if (x[5] < -2.03606114e-01f) {  /* LOG */
                if (x[2] < 1.85896552e+00f) {  /* WAMP */
                    return 2.23384681e-03f;
                } else {
                    return 7.60546476e-02f;
                }
            } else {
                if (x[5] < -8.31438527e-02f) {  /* LOG */
                    return -8.98728594e-02f;
                } else {
                    return -6.40646729e-04f;
                }
            }
        } else {
            if (x[1] < -3.61033753e-02f) {  /* WL */
                if (x[2] < 7.91900337e-01f) {  /* WAMP */
                    return -1.16481245e-01f;
                } else {
                    return 2.29984857e-02f;
                }
            } else {
                if (x[13] < -6.22344434e-01f) {  /* SpectralKurtosis */
                    return 3.81744206e-02f;
                } else {
                    return 1.02483273e-01f;
                }
            }
        }
    } else {
        if (x[6] < -3.62258814e-02f) {  /* SSI */
            if (x[2] < 7.65223742e-01f) {  /* WAMP */
                if (x[1] < -3.66576836e-02f) {  /* WL */
                    return -1.70264603e-03f;
                } else {
                    return 8.97895824e-03f;
                }
            } else {
                if (x[12] < -6.73046038e-02f) {  /* SpectralEntropy */
                    return -8.26483816e-02f;
                } else {
                    return -6.92604575e-03f;
                }
            }
        } else {
            if (x[4] < 4.07932818e-01f) {  /* SNR */
                return -9.43170711e-02f;
            } else {
                return -8.40332173e-03f;
            }
        }
    }
}

static inline float xgb_tree_172(const float *x) {
    if (x[14] < -4.24311534e-02f) {  /* MeanFreq */
        if (x[10] < -4.88750152e-02f) {  /* R_Mid */
            if (x[10] < -4.93235514e-02f) {  /* R_Mid */
                if (x[14] < -4.73180376e-02f) {  /* MeanFreq */
                    return 5.95564954e-04f;
                } else {
                    return 3.89273316e-02f;
                }
            } else {
                if (x[14] < -4.67843488e-02f) {  /* MeanFreq */
                    return -8.00547656e-03f;
                } else {
                    return -1.12252966e-01f;
                }
            }
        } else {
            if (x[5] < -8.38379562e-01f) {  /* LOG */
                if (x[8] < -3.62566933e-02f) {  /* E_Total */
                    return -2.89096637e-03f;
                } else {
                    return -6.02267571e-02f;
                }
            } else {
                if (x[1] < -3.71643864e-02f) {  /* WL */
                    return -5.97453304e-02f;
                } else {
                    return 1.40428608e-02f;
                }
            }
        }
    } else {
        if (x[4] < -1.41058254e+00f) {  /* SNR */
            if (x[14] < -4.06660475e-02f) {  /* MeanFreq */
                if (x[1] < -3.58238406e-02f) {  /* WL */
                    return -1.45340264e-02f;
                } else {
                    return 8.66326988e-02f;
                }
            } else {
                if (x[6] < -3.62601541e-02f) {  /* SSI */
                    return 5.72694503e-02f;
                } else {
                    return -1.71244256e-02f;
                }
            }
        } else {
            if (x[10] < -4.66369651e-02f) {  /* R_Mid */
                if (x[2] < 1.45881605e+00f) {  /* WAMP */
                    return -7.07669854e-02f;
                } else {
                    return 6.13185763e-02f;
                }
            } else {
                if (x[5] < -4.77476269e-02f) {  /* LOG */
                    return -8.69344398e-02f;
                } else {
                    return -7.96278659e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_173(const float *x) {
    if (x[2] < 1.53884590e+00f) {  /* WAMP */
        if (x[10] < -4.26135026e-02f) {  /* R_Mid */
            if (x[10] < -4.29179706e-02f) {  /* R_Mid */
                if (x[4] < -1.41058254e+00f) {  /* SNR */
                    return -5.00301011e-02f;
                } else {
                    return -7.26921658e-04f;
                }
            } else {
                if (x[4] < -1.12111843e+00f) {  /* SNR */
                    return 8.82568490e-03f;
                } else {
                    return -1.28301367e-01f;
                }
            }
        } else {
            if (x[4] < -1.65028071e+00f) {  /* SNR */
                if (x[2] < 1.32543290e+00f) {  /* WAMP */
                    return -2.99964752e-03f;
                } else {
                    return -9.82807651e-02f;
                }
            } else {
                if (x[3] < -3.62439677e-02f) {  /* VAR */
                    return 1.29526258e-02f;
                } else {
                    return 5.52272499e-02f;
                }
            }
        }
    } else {
        if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
            if (x[2] < 1.61887574e+00f) {  /* WAMP */
                if (x[4] < -9.49942052e-01f) {  /* SNR */
                    return -6.68232366e-02f;
                } else {
                    return 6.05330952e-02f;
                }
            } else {
                if (x[10] < -4.73016351e-02f) {  /* R_Mid */
                    return 7.02634603e-02f;
                } else {
                    return 5.49078127e-03f;
                }
            }
        } else {
            if (x[7] < -3.51352133e-02f) {  /* TFR_MeanEnergy */
                if (x[10] < -4.59087491e-02f) {  /* R_Mid */
                    return -6.31464412e-03f;
                } else {
                    return -1.00271083e-01f;
                }
            } else {
                if (x[10] < -4.59087491e-02f) {  /* R_Mid */
                    return -5.31458743e-02f;
                } else {
                    return 2.57593133e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_174(const float *x) {
    if (x[2] < -7.01990664e-01f) {  /* WAMP */
        if (x[5] < -3.13272268e-01f) {  /* LOG */
            if (x[1] < -3.73861082e-02f) {  /* WL */
                if (x[8] < -3.62487845e-02f) {  /* E_Total */
                    return 4.38219495e-02f;
                } else {
                    return -4.04725522e-02f;
                }
            } else {
                if (x[2] < -8.88727069e-01f) {  /* WAMP */
                    return -4.78807390e-02f;
                } else {
                    return 1.74671400e-03f;
                }
            }
        } else {
            if (x[1] < -3.73915508e-02f) {  /* WL */
                if (x[13] < 1.66717982e+00f) {  /* SpectralKurtosis */
                    return -1.92989521e-02f;
                } else {
                    return 6.79899240e-03f;
                }
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 3.88269387e-02f;
                } else {
                    return 3.26152146e-03f;
                }
            }
        }
    } else {
        if (x[12] < -9.31779444e-02f) {  /* SpectralEntropy */
            if (x[10] < -4.98902127e-02f) {  /* R_Mid */
                if (x[7] < -3.51401605e-02f) {  /* TFR_MeanEnergy */
                    return -8.63718335e-03f;
                } else {
                    return -5.84652126e-02f;
                }
            } else {
                return 5.84854148e-02f;
            }
        } else {
            if (x[10] < -5.01654521e-02f) {  /* R_Mid */
                if (x[13] < -5.95062017e-01f) {  /* SpectralKurtosis */
                    return -6.82415515e-02f;
                } else {
                    return -7.73796206e-03f;
                }
            } else {
                if (x[12] < -9.11695585e-02f) {  /* SpectralEntropy */
                    return 7.38662332e-02f;
                } else {
                    return 7.26878119e-04f;
                }
            }
        }
    }
}

static inline float xgb_tree_175(const float *x) {
    if (x[0] < -5.21450303e-02f) {  /* MAV */
        if (x[9] < -3.63057256e-02f) {  /* E_Low */
            if (x[1] < -3.74514051e-02f) {  /* WL */
                if (x[13] < 9.49356616e-01f) {  /* SpectralKurtosis */
                    return 8.74781832e-02f;
                } else {
                    return -2.90596876e-02f;
                }
            } else {
                if (x[2] < -7.82020569e-01f) {  /* WAMP */
                    return -2.51396578e-02f;
                } else {
                    return 8.76117032e-03f;
                }
            }
        } else {
            if (x[4] < -8.33197415e-01f) {  /* SNR */
                if (x[3] < -3.62439677e-02f) {  /* VAR */
                    return -1.48317397e-01f;
                } else {
                    return 3.67267393e-02f;
                }
            } else {
                if (x[0] < -5.31583168e-02f) {  /* MAV */
                    return 7.60281309e-02f;
                } else {
                    return 2.64465306e-02f;
                }
            }
        }
    } else {
        if (x[7] < -3.51415984e-02f) {  /* TFR_MeanEnergy */
            if (x[1] < -3.73038128e-02f) {  /* WL */
                if (x[1] < -3.73833887e-02f) {  /* WL */
                    return -1.51460050e-02f;
                } else {
                    return -8.16134140e-02f;
                }
            } else {
                if (x[10] < -5.07123880e-02f) {  /* R_Mid */
                    return 1.14777878e-01f;
                } else {
                    return -1.13170622e-02f;
                }
            }
        } else {
            if (x[2] < -7.01990664e-01f) {  /* WAMP */
                if (x[0] < -3.75898033e-02f) {  /* MAV */
                    return 8.98305792e-03f;
                } else {
                    return -8.77991971e-03f;
                }
            } else {
                if (x[14] < -5.03199585e-02f) {  /* MeanFreq */
                    return -4.72672731e-02f;
                } else {
                    return -2.28834734e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_176(const float *x) {
    if (x[10] < -5.05954102e-02f) {  /* R_Mid */
        if (x[12] < -8.05117339e-02f) {  /* SpectralEntropy */
            if (x[0] < -2.70186439e-02f) {  /* MAV */
                if (x[0] < -3.49031799e-02f) {  /* MAV */
                    return 2.26771529e-03f;
                } else {
                    return 3.08626778e-02f;
                }
            } else {
                if (x[8] < -3.62361595e-02f) {  /* E_Total */
                    return -8.36818144e-02f;
                } else {
                    return 2.61486266e-02f;
                }
            }
        } else {
            if (x[2] < -4.88577694e-01f) {  /* WAMP */
                if (x[10] < -5.07584922e-02f) {  /* R_Mid */
                    return 3.34386192e-02f;
                } else {
                    return 1.37050211e-01f;
                }
            } else {
                if (x[1] < -3.69780324e-02f) {  /* WL */
                    return -5.18350676e-02f;
                } else {
                    return 7.23996982e-02f;
                }
            }
        }
    } else {
        if (x[10] < -5.05524687e-02f) {  /* R_Mid */
            if (x[4] < 5.01608133e-01f) {  /* SNR */
                if (x[1] < -3.68970931e-02f) {  /* WL */
                    return -6.29192516e-02f;
                } else {
                    return 2.80478932e-02f;
                }
            } else {
                if (x[1] < -3.72970141e-02f) {  /* WL */
                    return -3.27870063e-02f;
                } else {
                    return 5.44623397e-02f;
                }
            }
        } else {
            if (x[14] < -4.99833450e-02f) {  /* MeanFreq */
                if (x[4] < 7.55880252e-02f) {  /* SNR */
                    return 5.93020432e-02f;
                } else {
                    return 4.58651967e-03f;
                }
            } else {
                if (x[10] < -5.01275882e-02f) {  /* R_Mid */
                    return -1.61489528e-02f;
                } else {
                    return -8.33500933e-04f;
                }
            }
        }
    }
}

static inline float xgb_tree_177(const float *x) {
    if (x[12] < -7.68788531e-02f) {  /* SpectralEntropy */
        if (x[13] < -5.51105559e-01f) {  /* SpectralKurtosis */
            return 1.31406173e-01f;
        } else {
            if (x[12] < -8.05117339e-02f) {  /* SpectralEntropy */
                if (x[14] < -4.97429892e-02f) {  /* MeanFreq */
                    return 6.79867459e-04f;
                } else {
                    return -3.73872034e-02f;
                }
            } else {
                if (x[7] < -3.51379141e-02f) {  /* TFR_MeanEnergy */
                    return 3.23905274e-02f;
                } else {
                    return -2.32919119e-02f;
                }
            }
        }
    } else {
        if (x[12] < -7.65110329e-02f) {  /* SpectralEntropy */
            if (x[7] < -3.51389386e-02f) {  /* TFR_MeanEnergy */
                if (x[14] < -4.96601611e-02f) {  /* MeanFreq */
                    return -2.13478655e-02f;
                } else {
                    return -1.44892320e-01f;
                }
            } else {
                if (x[0] < -3.11116558e-02f) {  /* MAV */
                    return 2.89499592e-02f;
                } else {
                    return -7.04350471e-02f;
                }
            }
        } else {
            if (x[13] < -4.01386201e-01f) {  /* SpectralKurtosis */
                if (x[1] < -3.68528850e-02f) {  /* WL */
                    return -2.55167075e-02f;
                } else {
                    return -1.68463145e-03f;
                }
            } else {
                if (x[10] < -5.06273359e-02f) {  /* R_Mid */
                    return -4.07276228e-02f;
                } else {
                    return 7.19299242e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_178(const float *x) {
    if (x[10] < -5.06605767e-02f) {  /* R_Mid */
        if (x[12] < -7.44713619e-02f) {  /* SpectralEntropy */
            if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                    return -2.98687606e-03f;
                } else {
                    return -2.37792544e-02f;
                }
            } else {
                if (x[12] < -9.68171358e-02f) {  /* SpectralEntropy */
                    return -7.20800785e-03f;
                } else {
                    return 3.16702574e-02f;
                }
            }
        } else {
            if (x[0] < -4.76828441e-02f) {  /* MAV */
                return 1.97659340e-02f;
            } else {
                return 1.29430071e-01f;
            }
        }
    } else {
        if (x[14] < -5.04442044e-02f) {  /* MeanFreq */
            if (x[0] < -4.31914292e-02f) {  /* MAV */
                return 1.05078490e-02f;
            } else {
                if (x[4] < 1.12795758e+00f) {  /* SNR */
                    return -2.23050434e-02f;
                } else {
                    return -1.01782851e-01f;
                }
            }
        } else {
            if (x[10] < -5.06273359e-02f) {  /* R_Mid */
                if (x[6] < -3.62456404e-02f) {  /* SSI */
                    return -1.24549111e-02f;
                } else {
                    return 3.02201845e-02f;
                }
            } else {
                if (x[14] < -4.88973148e-02f) {  /* MeanFreq */
                    return 4.21627564e-03f;
                } else {
                    return -1.55397260e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_179(const float *x) {
    if (x[5] < -3.40174288e-01f) {  /* LOG */
        if (x[1] < -3.69236171e-02f) {  /* WL */
            if (x[2] < -7.82020569e-01f) {  /* WAMP */
                if (x[1] < -3.71929519e-02f) {  /* WL */
                    return -9.00000613e-03f;
                } else {
                    return -8.23302716e-02f;
                }
            } else {
                if (x[13] < -5.19735336e-01f) {  /* SpectralKurtosis */
                    return -2.55134143e-02f;
                } else {
                    return 1.76017340e-02f;
                }
            }
        } else {
            if (x[2] < 4.45104212e-01f) {  /* WAMP */
                if (x[1] < -3.65386605e-02f) {  /* WL */
                    return -3.07454467e-02f;
                } else {
                    return -1.19658045e-01f;
                }
            } else {
                if (x[2] < 4.71780837e-01f) {  /* WAMP */
                    return 8.44557360e-02f;
                } else {
                    return -4.40678850e-04f;
                }
            }
        }
    } else {
        if (x[1] < -3.74037959e-02f) {  /* WL */
            if (x[2] < -9.42080319e-01f) {  /* WAMP */
                if (x[6] < -3.62446941e-02f) {  /* SSI */
                    return -6.90765819e-03f;
                } else {
                    return 2.72102747e-02f;
                }
            } else {
                if (x[0] < -4.41422984e-02f) {  /* MAV */
                    return 6.56359941e-02f;
                } else {
                    return -4.80551533e-02f;
                }
            }
        } else {
            if (x[1] < -3.73616256e-02f) {  /* WL */
                if (x[2] < -8.62050474e-01f) {  /* WAMP */
                    return 3.23249511e-02f;
                } else {
                    return -2.52951439e-02f;
                }
            } else {
                if (x[14] < -5.04630543e-02f) {  /* MeanFreq */
                    return 7.22264573e-02f;
                } else {
                    return 3.76688177e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_180(const float *x) {
    if (x[3] < -3.62430066e-02f) {  /* VAR */
        if (x[3] < -3.62434611e-02f) {  /* VAR */
            if (x[3] < -3.62435430e-02f) {  /* VAR */
                if (x[1] < -3.43962349e-02f) {  /* WL */
                    return -2.79856700e-04f;
                } else {
                    return -1.08648315e-01f;
                }
            } else {
                if (x[2] < 2.12573171e+00f) {  /* WAMP */
                    return 7.73012564e-02f;
                } else {
                    return -4.48176153e-02f;
                }
            }
        } else {
            if (x[2] < 2.12573171e+00f) {  /* WAMP */
                if (x[1] < -3.43962349e-02f) {  /* WL */
                    return -1.00730240e-01f;
                } else {
                    return -1.16525181e-02f;
                }
            } else {
                if (x[0] < -1.05430773e-02f) {  /* MAV */
                    return 5.75573146e-02f;
                } else {
                    return -2.85184886e-02f;
                }
            }
        }
    } else {
        if (x[8] < -3.61802876e-02f) {  /* E_Total */
            if (x[4] < -2.11604714e+00f) {  /* SNR */
                return 1.01333372e-01f;
            } else {
                if (x[1] < -3.45213786e-02f) {  /* WL */
                    return -4.96419705e-02f;
                } else {
                    return 6.72792718e-02f;
                }
            }
        } else {
            if (x[5] < 1.33196819e+00f) {  /* LOG */
                if (x[2] < -5.68607569e-01f) {  /* WAMP */
                    return -5.52508906e-02f;
                } else {
                    return 6.18518554e-02f;
                }
            } else {
                return -6.16844893e-02f;
            }
        }
    }
}

static inline float xgb_tree_181(const float *x) {
    if (x[3] < -3.62434611e-02f) {  /* VAR */
        if (x[5] < -2.83600867e-01f) {  /* LOG */
            if (x[10] < -5.07498011e-02f) {  /* R_Mid */
                if (x[2] < -9.68756974e-01f) {  /* WAMP */
                    return -3.57248150e-02f;
                } else {
                    return 7.03441724e-02f;
                }
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return -2.23797131e-02f;
                } else {
                    return -4.34806524e-03f;
                }
            }
        } else {
            if (x[9] < -3.62753011e-02f) {  /* E_Low */
                if (x[2] < -9.42080319e-01f) {  /* WAMP */
                    return -2.49618962e-02f;
                } else {
                    return 5.77579997e-02f;
                }
            } else {
                if (x[14] < -3.60739417e-02f) {  /* MeanFreq */
                    return 2.34204414e-03f;
                } else {
                    return 8.21684003e-02f;
                }
            }
        }
    } else {
        if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
            if (x[0] < -1.46995031e-03f) {  /* MAV */
                if (x[14] < -5.23453020e-03f) {  /* MeanFreq */
                    return -9.13043544e-02f;
                } else {
                    return 4.38930951e-02f;
                }
            } else {
                if (x[1] < -3.42500024e-02f) {  /* WL */
                    return 9.09373015e-02f;
                } else {
                    return 1.07347229e-02f;
                }
            }
        } else {
            if (x[5] < -1.17321730e-01f) {  /* LOG */
                if (x[6] < -3.62454951e-02f) {  /* SSI */
                    return -3.86048704e-02f;
                } else {
                    return 5.82669191e-02f;
                }
            } else {
                if (x[1] < -3.48002315e-02f) {  /* WL */
                    return -1.34775266e-01f;
                } else {
                    return -4.90325354e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_182(const float *x) {
    if (x[0] < -8.48704129e-02f) {  /* MAV */
        if (x[3] < -3.62441204e-02f) {  /* VAR */
            if (x[1] < -3.73480208e-02f) {  /* WL */
                if (x[12] < -9.47066844e-02f) {  /* SpectralEntropy */
                    return 4.14225645e-02f;
                } else {
                    return 1.74058199e-01f;
                }
            } else {
                if (x[13] < -3.95395875e-01f) {  /* SpectralKurtosis */
                    return 7.57244900e-02f;
                } else {
                    return -8.07675347e-02f;
                }
            }
        } else {
            if (x[1] < -3.72133590e-02f) {  /* WL */
                if (x[3] < -3.62440832e-02f) {  /* VAR */
                    return -5.68051115e-02f;
                } else {
                    return 5.51622622e-02f;
                }
            } else {
                if (x[12] < -7.52863139e-02f) {  /* SpectralEntropy */
                    return 1.21787481e-01f;
                } else {
                    return 1.24294860e-02f;
                }
            }
        }
    } else {
        if (x[1] < -3.53484228e-02f) {  /* WL */
            if (x[2] < 1.59219909e+00f) {  /* WAMP */
                if (x[3] < -3.62437628e-02f) {  /* VAR */
                    return -7.91270213e-05f;
                } else {
                    return -3.37962545e-02f;
                }
            } else {
                if (x[12] < -4.53857295e-02f) {  /* SpectralEntropy */
                    return -8.66918787e-02f;
                } else {
                    return -1.58803333e-02f;
                }
            }
        } else {
            if (x[6] < -3.62477340e-02f) {  /* SSI */
                if (x[2] < 1.75225902e+00f) {  /* WAMP */
                    return -1.95994228e-01f;
                } else {
                    return 6.86468035e-02f;
                }
            } else {
                if (x[10] < -4.95533496e-02f) {  /* R_Mid */
                    return -8.01827908e-02f;
                } else {
                    return 2.56078877e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_183(const float *x) {
    if (x[14] < -4.20331433e-02f) {  /* MeanFreq */
        if (x[10] < -4.82886098e-02f) {  /* R_Mid */
            if (x[1] < -3.65733504e-02f) {  /* WL */
                if (x[6] < -3.62458192e-02f) {  /* SSI */
                    return -5.93171967e-03f;
                } else {
                    return 3.19080730e-03f;
                }
            } else {
                if (x[12] < -5.81776053e-02f) {  /* SpectralEntropy */
                    return -5.05444966e-02f;
                } else {
                    return -2.73528183e-03f;
                }
            }
        } else {
            if (x[13] < -6.08117580e-01f) {  /* SpectralKurtosis */
                if (x[3] < -3.62437032e-02f) {  /* VAR */
                    return 3.84927206e-02f;
                } else {
                    return -9.75564495e-02f;
                }
            } else {
                if (x[14] < -4.59191911e-02f) {  /* MeanFreq */
                    return 1.13583794e-02f;
                } else {
                    return -1.56029779e-02f;
                }
            }
        }
    } else {
        if (x[10] < -4.95262817e-02f) {  /* R_Mid */
            if (x[12] < 1.59623742e-01f) {  /* SpectralEntropy */
                if (x[10] < -4.97956462e-02f) {  /* R_Mid */
                    return -3.32489312e-02f;
                } else {
                    return -1.35380059e-01f;
                }
            } else {
                return -4.73245932e-03f;
            }
        } else {
            if (x[10] < -4.85919639e-02f) {  /* R_Mid */
                if (x[0] < -3.94476987e-02f) {  /* MAV */
                    return 1.26741473e-02f;
                } else {
                    return 8.69229138e-02f;
                }
            } else {
                if (x[1] < -3.40786092e-02f) {  /* WL */
                    return -1.97897349e-02f;
                } else {
                    return 2.51981318e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_184(const float *x) {
    if (x[13] < -6.09963298e-01f) {  /* SpectralKurtosis */
        if (x[14] < -4.55845259e-02f) {  /* MeanFreq */
            if (x[10] < -4.91329283e-02f) {  /* R_Mid */
                if (x[10] < -4.92600910e-02f) {  /* R_Mid */
                    return 2.08948553e-02f;
                } else {
                    return -1.22376658e-01f;
                }
            } else {
                if (x[7] < -3.51424664e-02f) {  /* TFR_MeanEnergy */
                    return 2.72668637e-02f;
                } else {
                    return 9.99328569e-02f;
                }
            }
        } else {
            if (x[10] < -4.94689420e-02f) {  /* R_Mid */
                if (x[14] < -4.42967936e-02f) {  /* MeanFreq */
                    return -8.36197883e-02f;
                } else {
                    return 5.43027697e-03f;
                }
            } else {
                if (x[5] < 1.04134607e+00f) {  /* LOG */
                    return 1.31521598e-02f;
                } else {
                    return -2.23319326e-02f;
                }
            }
        }
    } else {
        if (x[13] < -6.04469478e-01f) {  /* SpectralKurtosis */
            if (x[3] < -3.62439789e-02f) {  /* VAR */
                if (x[3] < -3.62440161e-02f) {  /* VAR */
                    return -9.77570284e-03f;
                } else {
                    return -1.17649890e-01f;
                }
            } else {
                if (x[10] < -4.22207415e-02f) {  /* R_Mid */
                    return -2.56874394e-02f;
                } else {
                    return 3.93394753e-02f;
                }
            }
        } else {
            if (x[5] < 2.17295122e+00f) {  /* LOG */
                if (x[7] < -3.51292193e-02f) {  /* TFR_MeanEnergy */
                    return -6.92865753e-04f;
                } else {
                    return 3.01524382e-02f;
                }
            } else {
                if (x[3] < -3.62441055e-02f) {  /* VAR */
                    return 2.34960625e-03f;
                } else {
                    return -8.43976364e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_185(const float *x) {
    if (x[14] < -5.01721092e-02f) {  /* MeanFreq */
        if (x[10] < -4.98902127e-02f) {  /* R_Mid */
            if (x[5] < -2.05042243e+00f) {  /* LOG */
                if (x[10] < -5.03716618e-02f) {  /* R_Mid */
                    return 1.15605488e-01f;
                } else {
                    return -1.47374859e-02f;
                }
            } else {
                if (x[10] < -5.06735072e-02f) {  /* R_Mid */
                    return -3.79728037e-03f;
                } else {
                    return 7.94832502e-03f;
                }
            }
        } else {
            if (x[0] < -3.95021550e-02f) {  /* MAV */
                return 1.36030540e-01f;
            } else {
                if (x[1] < -3.71684693e-02f) {  /* WL */
                    return -4.20406125e-02f;
                } else {
                    return 1.18984632e-01f;
                }
            }
        }
    } else {
        if (x[14] < -5.00740409e-02f) {  /* MeanFreq */
            if (x[10] < -5.06124273e-02f) {  /* R_Mid */
                if (x[7] < -3.51418890e-02f) {  /* TFR_MeanEnergy */
                    return -9.32213441e-02f;
                } else {
                    return 2.18111258e-02f;
                }
            } else {
                if (x[1] < -3.71419415e-02f) {  /* WL */
                    return -1.34693552e-02f;
                } else {
                    return -4.24700156e-02f;
                }
            }
        } else {
            if (x[14] < -4.99593318e-02f) {  /* MeanFreq */
                if (x[14] < -4.99913245e-02f) {  /* MeanFreq */
                    return 1.20065140e-03f;
                } else {
                    return 4.07205522e-02f;
                }
            } else {
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return -1.79894157e-02f;
                } else {
                    return -1.85534940e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_186(const float *x) {
    if (x[10] < -5.04248068e-02f) {  /* R_Mid */
        if (x[6] < -3.62495929e-02f) {  /* SSI */
            if (x[1] < -3.74235176e-02f) {  /* WL */
                if (x[5] < -1.74581444e+00f) {  /* LOG */
                    return 9.37552378e-02f;
                } else {
                    return -1.74876209e-02f;
                }
            } else {
                if (x[2] < -4.35224444e-01f) {  /* WAMP */
                    return -8.85318071e-02f;
                } else {
                    return 2.97290981e-02f;
                }
            }
        } else {
            if (x[8] < -3.62539366e-02f) {  /* E_Total */
                if (x[1] < -3.73779461e-02f) {  /* WL */
                    return -4.07982022e-02f;
                } else {
                    return 1.03579402e-01f;
                }
            } else {
                if (x[7] < -3.51452604e-02f) {  /* TFR_MeanEnergy */
                    return -8.71366635e-02f;
                } else {
                    return -2.63929996e-03f;
                }
            }
        }
    } else {
        if (x[14] < -5.02355546e-02f) {  /* MeanFreq */
            if (x[7] < -3.51389758e-02f) {  /* TFR_MeanEnergy */
                if (x[7] < -3.51420566e-02f) {  /* TFR_MeanEnergy */
                    return 6.54149950e-02f;
                } else {
                    return -5.12652006e-03f;
                }
            } else {
                if (x[2] < -8.08697224e-01f) {  /* WAMP */
                    return 2.28838176e-02f;
                } else {
                    return 1.61994427e-01f;
                }
            }
        } else {
            if (x[14] < -5.01016416e-02f) {  /* MeanFreq */
                if (x[1] < -3.72677669e-02f) {  /* WL */
                    return -7.17277601e-02f;
                } else {
                    return -1.07997116e-02f;
                }
            } else {
                if (x[14] < -4.99593318e-02f) {  /* MeanFreq */
                    return 2.80643739e-02f;
                } else {
                    return 1.40883250e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_187(const float *x) {
    if (x[6] < -3.62454318e-02f) {  /* SSI */
        if (x[2] < 7.91900337e-01f) {  /* WAMP */
            if (x[10] < -5.04164770e-02f) {  /* R_Mid */
                if (x[13] < 5.13806462e-01f) {  /* SpectralKurtosis */
                    return 1.93260536e-02f;
                } else {
                    return -6.04092842e-03f;
                }
            } else {
                if (x[7] < -3.51402126e-02f) {  /* TFR_MeanEnergy */
                    return -6.46296842e-03f;
                } else {
                    return 4.05986831e-02f;
                }
            }
        } else {
            if (x[13] < -5.98463833e-01f) {  /* SpectralKurtosis */
                if (x[2] < 1.05866659e+00f) {  /* WAMP */
                    return -3.82647291e-02f;
                } else {
                    return 2.09442470e-02f;
                }
            } else {
                if (x[5] < -5.85202038e-01f) {  /* LOG */
                    return 8.50071087e-02f;
                } else {
                    return 3.29729468e-02f;
                }
            }
        }
    } else {
        if (x[0] < -4.21993881e-02f) {  /* MAV */
            if (x[10] < -4.35475595e-02f) {  /* R_Mid */
                if (x[10] < -4.90278602e-02f) {  /* R_Mid */
                    return 2.15347018e-03f;
                } else {
                    return -1.67590469e-01f;
                }
            } else {
                if (x[1] < -3.52021940e-02f) {  /* WL */
                    return 7.80145451e-02f;
                } else {
                    return -3.03522740e-02f;
                }
            }
        } else {
            if (x[1] < -3.58789302e-02f) {  /* WL */
                if (x[14] < -4.40656655e-02f) {  /* MeanFreq */
                    return -2.49811355e-03f;
                } else {
                    return -1.12451427e-01f;
                }
            } else {
                if (x[13] < -6.01742804e-01f) {  /* SpectralKurtosis */
                    return 1.89763159e-02f;
                } else {
                    return -3.12383771e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_188(const float *x) {
    if (x[10] < -5.07498011e-02f) {  /* R_Mid */
        if (x[0] < -3.82060073e-02f) {  /* MAV */
            if (x[13] < 1.66717982e+00f) {  /* SpectralKurtosis */
                if (x[1] < -3.74201164e-02f) {  /* WL */
                    return -8.62461925e-02f;
                } else {
                    return -1.96718592e-02f;
                }
            } else {
                if (x[0] < -3.97146493e-02f) {  /* MAV */
                    return 2.31730659e-02f;
                } else {
                    return -2.36402284e-02f;
                }
            }
        } else {
            if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                if (x[4] < 1.14327848e+00f) {  /* SNR */
                    return 9.97613836e-03f;
                } else {
                    return -3.88398245e-02f;
                }
            } else {
                if (x[10] < -5.07731363e-02f) {  /* R_Mid */
                    return 5.07214665e-02f;
                } else {
                    return -1.77449472e-02f;
                }
            }
        }
    } else {
        if (x[1] < -3.74343991e-02f) {  /* WL */
            if (x[13] < 1.73085487e+00f) {  /* SpectralKurtosis */
                if (x[13] < 1.39467204e+00f) {  /* SpectralKurtosis */
                    return 3.17782313e-02f;
                } else {
                    return 1.16020598e-01f;
                }
            } else {
                if (x[13] < 2.10181284e+00f) {  /* SpectralKurtosis */
                    return -9.72799584e-02f;
                } else {
                    return 3.82677317e-02f;
                }
            }
        } else {
            if (x[1] < -3.73806693e-02f) {  /* WL */
                if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                    return -2.50265058e-02f;
                } else {
                    return 1.19550370e-01f;
                }
            } else {
                if (x[13] < 1.03015113e+00f) {  /* SpectralKurtosis */
                    return -5.56110812e-04f;
                } else {
                    return 6.88736215e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_189(const float *x) {
    if (x[7] < -3.51523831e-02f) {  /* TFR_MeanEnergy */
        if (x[10] < -4.88750152e-02f) {  /* R_Mid */
            if (x[1] < -3.69657874e-02f) {  /* WL */
                if (x[1] < -3.73371392e-02f) {  /* WL */
                    return 8.81808326e-02f;
                } else {
                    return -5.06173493e-03f;
                }
            } else {
                if (x[3] < -3.62440608e-02f) {  /* VAR */
                    return 1.50103316e-01f;
                } else {
                    return 8.37608613e-03f;
                }
            }
        } else {
            if (x[3] < -3.62440869e-02f) {  /* VAR */
                if (x[4] < -7.40554512e-01f) {  /* SNR */
                    return 3.32359746e-02f;
                } else {
                    return -5.71998172e-02f;
                }
            } else {
                if (x[1] < -3.73582244e-02f) {  /* WL */
                    return 1.22813143e-01f;
                } else {
                    return 1.04888072e-02f;
                }
            }
        }
    } else {
        if (x[4] < -1.08698773e+00f) {  /* SNR */
            if (x[4] < -1.12111843e+00f) {  /* SNR */
                if (x[11] < -2.78367192e-01f) {  /* TopFreq2 */
                    return 4.23911735e-02f;
                } else {
                    return -1.17726428e-02f;
                }
            } else {
                if (x[10] < -4.10061479e-02f) {  /* R_Mid */
                    return -1.08070873e-01f;
                } else {
                    return 3.94329056e-02f;
                }
            }
        } else {
            if (x[10] < -3.47048715e-02f) {  /* R_Mid */
                if (x[6] < -3.62515897e-02f) {  /* SSI */
                    return -3.05574443e-02f;
                } else {
                    return 1.38536957e-03f;
                }
            } else {
                if (x[10] < -2.43428741e-02f) {  /* R_Mid */
                    return 8.54323432e-02f;
                } else {
                    return -2.99369190e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_190(const float *x) {
    if (x[14] < -5.04774898e-02f) {  /* MeanFreq */
        if (x[6] < -3.62448730e-02f) {  /* SSI */
            if (x[7] < -3.51392217e-02f) {  /* TFR_MeanEnergy */
                if (x[10] < -5.08435890e-02f) {  /* R_Mid */
                    return 8.60448405e-02f;
                } else {
                    return 1.77836791e-02f;
                }
            } else {
                return 1.31748483e-01f;
            }
        } else {
            if (x[10] < -5.07870540e-02f) {  /* R_Mid */
                if (x[10] < -5.07916100e-02f) {  /* R_Mid */
                    return -8.51052254e-03f;
                } else {
                    return 1.06826559e-01f;
                }
            } else {
                if (x[10] < -5.07556573e-02f) {  /* R_Mid */
                    return -1.08786739e-01f;
                } else {
                    return -3.43674957e-03f;
                }
            }
        }
    } else {
        if (x[13] < 2.32244062e+00f) {  /* SpectralKurtosis */
            if (x[10] < -5.08264527e-02f) {  /* R_Mid */
                if (x[14] < -5.04416488e-02f) {  /* MeanFreq */
                    return -9.33238044e-02f;
                } else {
                    return 2.72287764e-02f;
                }
            } else {
                if (x[13] < 1.66717982e+00f) {  /* SpectralKurtosis */
                    return -3.49325594e-04f;
                } else {
                    return 2.84871198e-02f;
                }
            }
        } else {
            if (x[7] < -3.51389162e-02f) {  /* TFR_MeanEnergy */
                if (x[10] < -5.07706180e-02f) {  /* R_Mid */
                    return -9.93973911e-02f;
                } else {
                    return 2.51200646e-02f;
                }
            } else {
                if (x[0] < -3.74304391e-02f) {  /* MAV */
                    return 7.15392157e-02f;
                } else {
                    return -4.52002548e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_191(const float *x) {
    if (x[8] < -3.62517126e-02f) {  /* E_Total */
        if (x[7] < -3.51452604e-02f) {  /* TFR_MeanEnergy */
            if (x[11] < 1.50948599e-01f) {  /* TopFreq2 */
                if (x[1] < -3.69909517e-02f) {  /* WL */
                    return -2.25014403e-03f;
                } else {
                    return 2.63265874e-02f;
                }
            } else {
                if (x[4] < 2.44320072e-02f) {  /* SNR */
                    return -4.59254459e-02f;
                } else {
                    return 1.98694095e-02f;
                }
            }
        } else {
            if (x[14] < -3.75789218e-02f) {  /* MeanFreq */
                if (x[14] < -4.57632206e-02f) {  /* MeanFreq */
                    return 3.98783199e-02f;
                } else {
                    return 1.22226439e-01f;
                }
            } else {
                return -7.45759606e-02f;
            }
        }
    } else {
        if (x[5] < -6.27905130e-01f) {  /* LOG */
            if (x[1] < -3.66250388e-02f) {  /* WL */
                return -1.29100576e-01f;
            } else {
                if (x[14] < -4.66619544e-02f) {  /* MeanFreq */
                    return 7.86485821e-02f;
                } else {
                    return -5.51715009e-02f;
                }
            }
        } else {
            if (x[0] < -5.31583168e-02f) {  /* MAV */
                if (x[7] < -3.51439863e-02f) {  /* TFR_MeanEnergy */
                    return -7.43416995e-02f;
                } else {
                    return 5.88802584e-02f;
                }
            } else {
                if (x[1] < -3.75234969e-02f) {  /* WL */
                    return -5.34204021e-02f;
                } else {
                    return -1.23330893e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_192(const float *x) {
    if (x[10] < -4.89861071e-02f) {  /* R_Mid */
        if (x[1] < -3.69236171e-02f) {  /* WL */
            if (x[13] < -5.36018252e-01f) {  /* SpectralKurtosis */
                if (x[3] < -3.62440906e-02f) {  /* VAR */
                    return -2.91940733e-03f;
                } else {
                    return 1.19559668e-01f;
                }
            } else {
                if (x[6] < -3.62388417e-02f) {  /* SSI */
                    return 2.74709164e-04f;
                } else {
                    return 3.57490480e-02f;
                }
            }
        } else {
            if (x[5] < 4.50353891e-01f) {  /* LOG */
                if (x[6] < -3.62420604e-02f) {  /* SSI */
                    return -1.17926737e-02f;
                } else {
                    return 7.61818811e-02f;
                }
            } else {
                if (x[6] < -3.62384804e-02f) {  /* SSI */
                    return -5.11872880e-02f;
                } else {
                    return 1.69645268e-02f;
                }
            }
        }
    } else {
        if (x[13] < -5.22637427e-01f) {  /* SpectralKurtosis */
            if (x[13] < -5.67353427e-01f) {  /* SpectralKurtosis */
                if (x[3] < -3.62440683e-02f) {  /* VAR */
                    return 5.12414984e-02f;
                } else {
                    return 3.81227839e-03f;
                }
            } else {
                if (x[5] < 9.22445506e-02f) {  /* LOG */
                    return 2.49603181e-03f;
                } else {
                    return -3.10479738e-02f;
                }
            }
        } else {
            if (x[1] < -3.71643864e-02f) {  /* WL */
                if (x[13] < -5.06195843e-01f) {  /* SpectralKurtosis */
                    return 7.71179497e-02f;
                } else {
                    return -7.84198865e-02f;
                }
            } else {
                if (x[4] < -9.75628942e-02f) {  /* SNR */
                    return 5.62110431e-02f;
                } else {
                    return 1.00472523e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_193(const float *x) {
    if (x[13] < 3.29180932e+00f) {  /* SpectralKurtosis */
        if (x[14] < -5.04676141e-02f) {  /* MeanFreq */
            if (x[2] < -8.88727069e-01f) {  /* WAMP */
                if (x[1] < -3.74010727e-02f) {  /* WL */
                    return 2.11023446e-02f;
                } else {
                    return 9.81852338e-02f;
                }
            } else {
                if (x[14] < -5.04728109e-02f) {  /* MeanFreq */
                    return -2.82669850e-02f;
                } else {
                    return 5.99142127e-02f;
                }
            }
        } else {
            if (x[1] < -3.72582451e-02f) {  /* WL */
                if (x[1] < -3.72650474e-02f) {  /* WL */
                    return -2.39216094e-03f;
                } else {
                    return -6.01214170e-02f;
                }
            } else {
                if (x[13] < 3.09581488e-01f) {  /* SpectralKurtosis */
                    return 1.62138604e-03f;
                } else {
                    return 5.69348894e-02f;
                }
            }
        }
    } else {
        if (x[7] < -3.51393968e-02f) {  /* TFR_MeanEnergy */
            if (x[7] < -3.51398140e-02f) {  /* TFR_MeanEnergy */
                if (x[0] < -4.14676405e-02f) {  /* MAV */
                    return 1.85076296e-02f;
                } else {
                    return -9.97257456e-02f;
                }
            } else {
                if (x[1] < -3.74854095e-02f) {  /* WL */
                    return 1.09154023e-01f;
                } else {
                    return 8.21053516e-03f;
                }
            }
        } else {
            if (x[13] < 5.23936605e+00f) {  /* SpectralKurtosis */
                if (x[4] < 1.16324902e+00f) {  /* SNR */
                    return 3.28701735e-03f;
                } else {
                    return -8.06650147e-02f;
                }
            } else {
                if (x[1] < -3.75234969e-02f) {  /* WL */
                    return -4.08512317e-02f;
                } else {
                    return 6.61489964e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_194(const float *x) {
    if (x[1] < -3.15750279e-02f) {  /* WL */
        if (x[2] < -9.95433569e-01f) {  /* WAMP */
            if (x[1] < -3.73915508e-02f) {  /* WL */
                if (x[7] < -3.51391248e-02f) {  /* TFR_MeanEnergy */
                    return -5.41224517e-03f;
                } else {
                    return 2.95478757e-02f;
                }
            } else {
                if (x[0] < -4.53202799e-02f) {  /* MAV */
                    return -4.66204174e-02f;
                } else {
                    return 8.43775198e-02f;
                }
            }
        } else {
            if (x[13] < -5.88590384e-01f) {  /* SpectralKurtosis */
                if (x[4] < -6.92450643e-01f) {  /* SNR */
                    return 3.20824911e-03f;
                } else {
                    return 3.53420936e-02f;
                }
            } else {
                if (x[1] < -3.60816084e-02f) {  /* WL */
                    return 4.75370092e-04f;
                } else {
                    return -7.55728260e-02f;
                }
            }
        }
    } else {
        if (x[2] < 4.71780837e-01f) {  /* WAMP */
            return -1.60004571e-02f;
        } else {
            return -6.29373267e-02f;
        }
    }
}

static inline float xgb_tree_195(const float *x) {
    if (x[1] < -3.72854508e-02f) {  /* WL */
        if (x[2] < -8.08697224e-01f) {  /* WAMP */
            if (x[14] < -5.03934473e-02f) {  /* MeanFreq */
                if (x[1] < -3.73425819e-02f) {  /* WL */
                    return -3.13538103e-03f;
                } else {
                    return 4.51991260e-02f;
                }
            } else {
                if (x[14] < -5.03902994e-02f) {  /* MeanFreq */
                    return -9.53958854e-02f;
                } else {
                    return -1.21763907e-02f;
                }
            }
        } else {
            if (x[14] < -5.04527353e-02f) {  /* MeanFreq */
                if (x[4] < 1.52368701e+00f) {  /* SNR */
                    return 7.96277970e-02f;
                } else {
                    return -5.65744825e-02f;
                }
            } else {
                if (x[14] < -5.03840819e-02f) {  /* MeanFreq */
                    return -6.58648014e-02f;
                } else {
                    return -1.28690181e-02f;
                }
            }
        }
    } else {
        if (x[2] < -7.28667319e-01f) {  /* WAMP */
            if (x[14] < -5.03866337e-02f) {  /* MeanFreq */
                if (x[10] < -5.07706180e-02f) {  /* R_Mid */
                    return 1.92473475e-02f;
                } else {
                    return 1.38727829e-01f;
                }
            } else {
                if (x[5] < -7.97443569e-01f) {  /* LOG */
                    return -4.95666116e-02f;
                } else {
                    return 1.06484052e-02f;
                }
            }
        } else {
            if (x[4] < 6.16060555e-01f) {  /* SNR */
                if (x[14] < -5.03029414e-02f) {  /* MeanFreq */
                    return 6.61642477e-02f;
                } else {
                    return 5.45363757e-04f;
                }
            } else {
                if (x[0] < -4.63906750e-02f) {  /* MAV */
                    return 6.21165298e-02f;
                } else {
                    return -5.33578172e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_196(const float *x) {
    if (x[14] < -4.54153642e-02f) {  /* MeanFreq */
        if (x[12] < 1.65871251e-02f) {  /* SpectralEntropy */
            if (x[13] < -6.18349969e-01f) {  /* SpectralKurtosis */
                if (x[3] < -3.62439603e-02f) {  /* VAR */
                    return 7.83232898e-02f;
                } else {
                    return 1.27307391e-02f;
                }
            } else {
                if (x[14] < -4.55845259e-02f) {  /* MeanFreq */
                    return -3.58750083e-04f;
                } else {
                    return 6.39789626e-02f;
                }
            }
        } else {
            if (x[12] < 2.60020643e-02f) {  /* SpectralEntropy */
                if (x[4] < -1.12111843e+00f) {  /* SNR */
                    return 1.23104572e-01f;
                } else {
                    return 3.57277095e-02f;
                }
            } else {
                if (x[3] < -3.62438709e-02f) {  /* VAR */
                    return -5.99664561e-02f;
                } else {
                    return 6.14405535e-02f;
                }
            }
        }
    } else {
        if (x[4] < -8.78272831e-01f) {  /* SNR */
            if (x[1] < -3.63727100e-02f) {  /* WL */
                if (x[1] < -3.66916917e-02f) {  /* WL */
                    return -3.24319266e-02f;
                } else {
                    return 7.19055533e-02f;
                }
            } else {
                if (x[1] < -3.61033753e-02f) {  /* WL */
                    return -7.05909133e-02f;
                } else {
                    return -5.64686814e-03f;
                }
            }
        } else {
            if (x[12] < -2.36843526e-02f) {  /* SpectralEntropy */
                return 4.66519110e-02f;
            } else {
                return -1.31078586e-01f;
            }
        }
    }
}

static inline float xgb_tree_197(const float *x) {
    if (x[9] < -3.63977104e-02f) {  /* E_Low */
        if (x[1] < -3.73371392e-02f) {  /* WL */
            if (x[0] < -9.27947909e-02f) {  /* MAV */
                if (x[4] < 3.53109509e-01f) {  /* SNR */
                    return 3.58669646e-02f;
                } else {
                    return 1.52924880e-01f;
                }
            } else {
                return -5.25030456e-02f;
            }
        } else {
            if (x[1] < -3.72133590e-02f) {  /* WL */
                if (x[2] < -8.88727069e-01f) {  /* WAMP */
                    return -1.01272419e-01f;
                } else {
                    return -1.92425698e-02f;
                }
            } else {
                if (x[13] < -4.89954382e-01f) {  /* SpectralKurtosis */
                    return 7.75880599e-03f;
                } else {
                    return 1.02757707e-01f;
                }
            }
        }
    } else {
        if (x[0] < 4.67124060e-02f) {  /* MAV */
            if (x[3] < -3.62439491e-02f) {  /* VAR */
                if (x[2] < 1.83228886e+00f) {  /* WAMP */
                    return 1.00182754e-03f;
                } else {
                    return -6.90770894e-02f;
                }
            } else {
                if (x[3] < -3.62439267e-02f) {  /* VAR */
                    return -4.53650951e-02f;
                } else {
                    return -4.03743470e-03f;
                }
            }
        } else {
            if (x[1] < -3.56959701e-02f) {  /* WL */
                if (x[4] < 1.68187037e-01f) {  /* SNR */
                    return -9.10041705e-02f;
                } else {
                    return -1.93640161e-02f;
                }
            } else {
                if (x[11] < -6.37092888e-02f) {  /* TopFreq2 */
                    return 3.18662971e-02f;
                } else {
                    return -3.85988913e-02f;
                }
            }
        }
    }
}

static inline float xgb_tree_198(const float *x) {
    if (x[1] < -3.58789302e-02f) {  /* WL */
        if (x[12] < -3.23307924e-02f) {  /* SpectralEntropy */
            if (x[13] < -5.92814326e-01f) {  /* SpectralKurtosis */
                if (x[10] < -4.91329283e-02f) {  /* R_Mid */
                    return -9.69660748e-03f;
                } else {
                    return 5.87426014e-02f;
                }
            } else {
                if (x[2] < 1.19204974e+00f) {  /* WAMP */
                    return -6.86103420e-04f;
                } else {
                    return -3.82169746e-02f;
                }
            }
        } else {
            if (x[0] < -5.51570095e-02f) {  /* MAV */
                if (x[3] < -3.62439267e-02f) {  /* VAR */
                    return -1.64582785e-02f;
                } else {
                    return 4.46191765e-02f;
                }
            } else {
                if (x[3] < -3.62440012e-02f) {  /* VAR */
                    return -8.52524340e-02f;
                } else {
                    return -1.88753102e-02f;
                }
            }
        }
    } else {
        if (x[3] < -3.62440050e-02f) {  /* VAR */
            if (x[12] < -4.10436913e-02f) {  /* SpectralEntropy */
                return -5.88293304e-04f;
            } else {
                return -1.04735270e-01f;
            }
        } else {
            if (x[2] < 1.88564217e+00f) {  /* WAMP */
                if (x[3] < -3.62438001e-02f) {  /* VAR */
                    return 3.83145213e-02f;
                } else {
                    return 7.40700588e-03f;
                }
            } else {
                if (x[1] < -3.55293415e-02f) {  /* WL */
                    return -8.28641877e-02f;
                } else {
                    return 2.84114736e-03f;
                }
            }
        }
    }
}

static inline float xgb_tree_199(const float *x) {
    if (x[7] < -3.51361148e-02f) {  /* TFR_MeanEnergy */
        if (x[5] < 1.81614906e-01f) {  /* LOG */
            if (x[14] < -5.04676141e-02f) {  /* MeanFreq */
                if (x[10] < -5.07470667e-02f) {  /* R_Mid */
                    return 4.61888453e-03f;
                } else {
                    return 7.51554891e-02f;
                }
            } else {
                if (x[3] < -3.62437777e-02f) {  /* VAR */
                    return -2.90706544e-03f;
                } else {
                    return 1.79739054e-02f;
                }
            }
        } else {
            if (x[5] < 2.21134692e-01f) {  /* LOG */
                if (x[11] < 3.65606487e-01f) {  /* TopFreq2 */
                    return -4.86083440e-02f;
                } else {
                    return 3.59382443e-02f;
                }
            } else {
                if (x[12] < -7.56563991e-02f) {  /* SpectralEntropy */
                    return -2.92072780e-02f;
                } else {
                    return 3.09575778e-02f;
                }
            }
        }
    } else {
        if (x[12] < -9.26299021e-02f) {  /* SpectralEntropy */
            if (x[10] < -5.03106415e-02f) {  /* R_Mid */
                if (x[5] < 5.14328301e-01f) {  /* LOG */
                    return -8.96894187e-02f;
                } else {
                    return -2.64190044e-02f;
                }
            } else {
                if (x[6] < -3.62418666e-02f) {  /* SSI */
                    return -2.79240049e-02f;
                } else {
                    return 7.55915046e-02f;
                }
            }
        } else {
            if (x[14] < -5.02825454e-02f) {  /* MeanFreq */
                if (x[14] < -5.03358580e-02f) {  /* MeanFreq */
                    return -3.17567699e-02f;
                } else {
                    return 2.03517318e-01f;
                }
            } else {
                if (x[14] < -5.02071157e-02f) {  /* MeanFreq */
                    return -6.00484200e-02f;
                } else {
                    return 1.21608218e-02f;
                }
            }
        }
    }
}

static inline int predict(const float *x_all) {
    float xn[N_FEATURES];
    normalize_select(x_all, xn);
    float score = 0.0f;
    score += xgb_tree_0(xn);
    score += xgb_tree_1(xn);
    score += xgb_tree_2(xn);
    score += xgb_tree_3(xn);
    score += xgb_tree_4(xn);
    score += xgb_tree_5(xn);
    score += xgb_tree_6(xn);
    score += xgb_tree_7(xn);
    score += xgb_tree_8(xn);
    score += xgb_tree_9(xn);
    score += xgb_tree_10(xn);
    score += xgb_tree_11(xn);
    score += xgb_tree_12(xn);
    score += xgb_tree_13(xn);
    score += xgb_tree_14(xn);
    score += xgb_tree_15(xn);
    score += xgb_tree_16(xn);
    score += xgb_tree_17(xn);
    score += xgb_tree_18(xn);
    score += xgb_tree_19(xn);
    score += xgb_tree_20(xn);
    score += xgb_tree_21(xn);
    score += xgb_tree_22(xn);
    score += xgb_tree_23(xn);
    score += xgb_tree_24(xn);
    score += xgb_tree_25(xn);
    score += xgb_tree_26(xn);
    score += xgb_tree_27(xn);
    score += xgb_tree_28(xn);
    score += xgb_tree_29(xn);
    score += xgb_tree_30(xn);
    score += xgb_tree_31(xn);
    score += xgb_tree_32(xn);
    score += xgb_tree_33(xn);
    score += xgb_tree_34(xn);
    score += xgb_tree_35(xn);
    score += xgb_tree_36(xn);
    score += xgb_tree_37(xn);
    score += xgb_tree_38(xn);
    score += xgb_tree_39(xn);
    score += xgb_tree_40(xn);
    score += xgb_tree_41(xn);
    score += xgb_tree_42(xn);
    score += xgb_tree_43(xn);
    score += xgb_tree_44(xn);
    score += xgb_tree_45(xn);
    score += xgb_tree_46(xn);
    score += xgb_tree_47(xn);
    score += xgb_tree_48(xn);
    score += xgb_tree_49(xn);
    score += xgb_tree_50(xn);
    score += xgb_tree_51(xn);
    score += xgb_tree_52(xn);
    score += xgb_tree_53(xn);
    score += xgb_tree_54(xn);
    score += xgb_tree_55(xn);
    score += xgb_tree_56(xn);
    score += xgb_tree_57(xn);
    score += xgb_tree_58(xn);
    score += xgb_tree_59(xn);
    score += xgb_tree_60(xn);
    score += xgb_tree_61(xn);
    score += xgb_tree_62(xn);
    score += xgb_tree_63(xn);
    score += xgb_tree_64(xn);
    score += xgb_tree_65(xn);
    score += xgb_tree_66(xn);
    score += xgb_tree_67(xn);
    score += xgb_tree_68(xn);
    score += xgb_tree_69(xn);
    score += xgb_tree_70(xn);
    score += xgb_tree_71(xn);
    score += xgb_tree_72(xn);
    score += xgb_tree_73(xn);
    score += xgb_tree_74(xn);
    score += xgb_tree_75(xn);
    score += xgb_tree_76(xn);
    score += xgb_tree_77(xn);
    score += xgb_tree_78(xn);
    score += xgb_tree_79(xn);
    score += xgb_tree_80(xn);
    score += xgb_tree_81(xn);
    score += xgb_tree_82(xn);
    score += xgb_tree_83(xn);
    score += xgb_tree_84(xn);
    score += xgb_tree_85(xn);
    score += xgb_tree_86(xn);
    score += xgb_tree_87(xn);
    score += xgb_tree_88(xn);
    score += xgb_tree_89(xn);
    score += xgb_tree_90(xn);
    score += xgb_tree_91(xn);
    score += xgb_tree_92(xn);
    score += xgb_tree_93(xn);
    score += xgb_tree_94(xn);
    score += xgb_tree_95(xn);
    score += xgb_tree_96(xn);
    score += xgb_tree_97(xn);
    score += xgb_tree_98(xn);
    score += xgb_tree_99(xn);
    score += xgb_tree_100(xn);
    score += xgb_tree_101(xn);
    score += xgb_tree_102(xn);
    score += xgb_tree_103(xn);
    score += xgb_tree_104(xn);
    score += xgb_tree_105(xn);
    score += xgb_tree_106(xn);
    score += xgb_tree_107(xn);
    score += xgb_tree_108(xn);
    score += xgb_tree_109(xn);
    score += xgb_tree_110(xn);
    score += xgb_tree_111(xn);
    score += xgb_tree_112(xn);
    score += xgb_tree_113(xn);
    score += xgb_tree_114(xn);
    score += xgb_tree_115(xn);
    score += xgb_tree_116(xn);
    score += xgb_tree_117(xn);
    score += xgb_tree_118(xn);
    score += xgb_tree_119(xn);
    score += xgb_tree_120(xn);
    score += xgb_tree_121(xn);
    score += xgb_tree_122(xn);
    score += xgb_tree_123(xn);
    score += xgb_tree_124(xn);
    score += xgb_tree_125(xn);
    score += xgb_tree_126(xn);
    score += xgb_tree_127(xn);
    score += xgb_tree_128(xn);
    score += xgb_tree_129(xn);
    score += xgb_tree_130(xn);
    score += xgb_tree_131(xn);
    score += xgb_tree_132(xn);
    score += xgb_tree_133(xn);
    score += xgb_tree_134(xn);
    score += xgb_tree_135(xn);
    score += xgb_tree_136(xn);
    score += xgb_tree_137(xn);
    score += xgb_tree_138(xn);
    score += xgb_tree_139(xn);
    score += xgb_tree_140(xn);
    score += xgb_tree_141(xn);
    score += xgb_tree_142(xn);
    score += xgb_tree_143(xn);
    score += xgb_tree_144(xn);
    score += xgb_tree_145(xn);
    score += xgb_tree_146(xn);
    score += xgb_tree_147(xn);
    score += xgb_tree_148(xn);
    score += xgb_tree_149(xn);
    score += xgb_tree_150(xn);
    score += xgb_tree_151(xn);
    score += xgb_tree_152(xn);
    score += xgb_tree_153(xn);
    score += xgb_tree_154(xn);
    score += xgb_tree_155(xn);
    score += xgb_tree_156(xn);
    score += xgb_tree_157(xn);
    score += xgb_tree_158(xn);
    score += xgb_tree_159(xn);
    score += xgb_tree_160(xn);
    score += xgb_tree_161(xn);
    score += xgb_tree_162(xn);
    score += xgb_tree_163(xn);
    score += xgb_tree_164(xn);
    score += xgb_tree_165(xn);
    score += xgb_tree_166(xn);
    score += xgb_tree_167(xn);
    score += xgb_tree_168(xn);
    score += xgb_tree_169(xn);
    score += xgb_tree_170(xn);
    score += xgb_tree_171(xn);
    score += xgb_tree_172(xn);
    score += xgb_tree_173(xn);
    score += xgb_tree_174(xn);
    score += xgb_tree_175(xn);
    score += xgb_tree_176(xn);
    score += xgb_tree_177(xn);
    score += xgb_tree_178(xn);
    score += xgb_tree_179(xn);
    score += xgb_tree_180(xn);
    score += xgb_tree_181(xn);
    score += xgb_tree_182(xn);
    score += xgb_tree_183(xn);
    score += xgb_tree_184(xn);
    score += xgb_tree_185(xn);
    score += xgb_tree_186(xn);
    score += xgb_tree_187(xn);
    score += xgb_tree_188(xn);
    score += xgb_tree_189(xn);
    score += xgb_tree_190(xn);
    score += xgb_tree_191(xn);
    score += xgb_tree_192(xn);
    score += xgb_tree_193(xn);
    score += xgb_tree_194(xn);
    score += xgb_tree_195(xn);
    score += xgb_tree_196(xn);
    score += xgb_tree_197(xn);
    score += xgb_tree_198(xn);
    score += xgb_tree_199(xn);
    float prob = 1.0f / (1.0f + expf(-score));
    return (prob >= 0.5f) ? 1 : 0;
}

#endif /* MODEL_H */
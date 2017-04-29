// Comment / Uncomment lines in CMakeLists.txt that refer to *.cpp with main() functions
// then Uncomment next line
#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file.
//                          Once you have more than one file with unit tests in you'll just #include "catch.hpp" and go.
//                          https://github.com/philsquared/Catch/blob/master/docs/tutorial.md
#include "catch.hpp"

#include "Eigen/Dense"
#include "ukf.hpp"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * UKF test of generating sigma points.
 */

SCENARIO("UKF test of generating sigma points.",
         "[ukf_gen_sigma_pts]") {

    GIVEN("an empty 5 row x 11 col matrix exists") {
        //Create a UKF instance
        UKF ukf;

        MatrixXd Xsig = MatrixXd(11, 5);
        REQUIRE(Xsig.size() == 55);

        WHEN("GenerateSigmaPoints is called") {
            ukf.GenerateSigmaPoints(&Xsig);

            THEN("the results are correct") {
                //print result
                //cout << "Xsig(0,0) = " << Xsig(0,0) << endl;

                REQUIRE(Xsig(0,0) == 5.7441);
            }
        }
    }
}

/**
 * UKF test of augmented sigma points.
 */

SCENARIO("UKF test of augmented sigma points.",
         "[ukf_aug_sigma_pts]") {

    GIVEN("an empty 7 row x 15 col matrix exists") {
        //Create a UKF instance
        UKF ukf;

        MatrixXd Xsig_aug = MatrixXd(15, 7);
        REQUIRE(Xsig_aug.size() == 105);

        WHEN("AugmentedSigmaPoints is called") {
            ukf.AugmentedSigmaPoints(&Xsig_aug);

            THEN("the results are correct") {
                //REQUIRE(Xsig_aug(0,0) == 5.7441);  passes
                string s = to_string(Xsig_aug(6,14));  // "-0.346410"
                REQUIRE(s == "-0.346410");
            }
        }
    }
}

/**
 * UKF test of augmented sigma points.
 */

SCENARIO("UKF test of sigma point prediction.",
         "[ukf_sigma_pt_prediction]") {

    GIVEN("an empty 5 row x 15 col matrix exists") {
        //Create a UKF instance
        UKF ukf;

        MatrixXd Xsig_pred = MatrixXd(15, 5);  // aka 5 row x 15 col
        REQUIRE(Xsig_pred.size() == 75);


        WHEN("SigmaPointPrediction is called") {
            ukf.SigmaPointPrediction(&Xsig_pred);

            THEN("the results are correct") {
                string s = to_string(Xsig_pred(4,14));  // "0.318159"
                REQUIRE(s == "0.318159");
            }
        }
    }
}

/**
 * UKF test of augmented sigma points.
 */

SCENARIO("UKF test calc state mean and covariance of predicted sigma.",
         "[ukf_mean_and_covariance]") {

    GIVEN("an empty 5 row x 15 col matrix exists") {
        //Create a UKF instance
        UKF ukf;

        VectorXd x_pred = VectorXd(5);
        MatrixXd P_pred = MatrixXd(5, 5);  // constructor uses 5 col x 5 row params, math notation uses 5 row x 5 col
        REQUIRE(x_pred.size() == 5);
        REQUIRE(P_pred.size() == 25);


        WHEN("PredictMeanAndCovariance is called") {
            ukf.PredictMeanAndCovariance(&x_pred, &P_pred);

            THEN("the results are correct") {
                string xs = to_string(x_pred(0));  // "5.936373"
                REQUIRE(xs == "5.936373");
                string ps = to_string(P_pred(0,0));  // "0.005434"
                REQUIRE(ps == "0.005434");
            }
        }
    }
}

/**
 * UKF test of measurement prediction including sensor noise.
 */

SCENARIO("UKF test of measurement prediction including sensor noise.",
         "[ukf_measurement_prediction_incl_sensor_noise]") {

    GIVEN("an empty 3 row Vector 'z_out' and 3 col x 3 row matrix 'S_out' exists") {
        //Create a UKF instance
        UKF ukf;

        VectorXd z_out = VectorXd(3);
        MatrixXd S_out = MatrixXd(3, 3);

        REQUIRE(z_out.size() == 3);
        REQUIRE(S_out.size() == 9);


        WHEN("PredictRadarMeasurement is called") {
            ukf.PredictRadarMeasurement(&z_out, &S_out);

            THEN("the results are correct") {
                string zs = to_string(z_out(0));  // "6.121547"
                REQUIRE(zs == "6.121547");
                string ss = to_string(S_out(2,2));  // "0.018092"
                REQUIRE(ss == "0.018092");
            }
        }
    }
}

/**
 * UKF test of updating state, the final step in pipeline.
 */

SCENARIO("UKF test of updating state, the final step in pipeline.",
         "[ukf_update_state]") {

    GIVEN("an empty 5 row Vector 'x_out' and 5 col x 5 row matrix 'P_out' exists") {
        //Create a UKF instance
        UKF ukf;

        VectorXd x_out = VectorXd(5);
        MatrixXd P_out = MatrixXd(5, 5);

        REQUIRE(x_out.size() == 5);
        REQUIRE(P_out.size() == 25);

        WHEN("UpdateState is called") {
            ukf.UpdateState(&x_out, &P_out);

            THEN("the results are correct") {
                string xs = to_string(x_out(0));  // "5.922762"
                REQUIRE(xs == "5.922762");
                string ps = to_string(P_out(4,4));  // "0.008818"
                REQUIRE(ps == "0.008818");
            }
        }
    }
}
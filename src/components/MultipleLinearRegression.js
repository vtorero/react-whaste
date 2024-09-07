import { Matrix } from 'ml-matrix';
import { useState, useEffect } from 'react';
import MultivariateLinearRegression from 'ml-regression-multivariate-linear';
import { BASE_URL } from '../utils/Constants';

const MultipleLinearRegression = ({ data }) => {
  const [predictionDenormalized, setPredictionDenormalized] = useState(0);
  const [meanSquaredError, setMeanSquaredError] = useState(0);
  const [rSquared, setRSquared] = useState(0);

  useEffect(() => {
    const { offers, wastes, demands, sales } = data;

    // Ensure input arrays have the same length
    if ([offers, wastes, demands, sales].some(arr => arr.length !== offers.length)) {
      console.error('Input arrays must have the same length');
      return;
    }

    // Helper function to normalize data
    const normalize = (arr) => {
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      return arr.map(value => (value - min) / (max - min));
    };

    const minMaxScale = (value, min, max) => (value - min) / (max - min);
    const denormalize = (normValue, min, max) => normValue * (max - min) + min;

    // Normalize input data
    const normOffers = normalize(offers);
    const normWastes = normalize(wastes);
    const normDemands = normalize(demands);
    const normSales = normalize(sales);

    console.log('Normalized offers:', normOffers);
    console.log('Normalized wastes:', normWastes);
    console.log('Normalized demands:', normDemands);
    console.log('Normalized sales:', normSales);

    // Prepare input matrix for training (X) and output matrix (Y)
    const X = new Matrix([normOffers, normWastes, normDemands]).transpose();
    const Y = new Matrix([normSales]).transpose();

    console.log('Matrix X dimensions:', X.rows, X.columns);
    console.log('Matrix Y dimensions:', Y.rows, Y.columns);

    // Train-test split: Use 80% for training and 20% for testing
    const trainTestSplit = 0.8;
    const splitIndex = Math.floor(X.rows * trainTestSplit);

    if (splitIndex === 0 || splitIndex >= X.rows) {
      console.error('Train-test split resulted in empty or invalid matrices');
      return;
    }

    const X_train = X.subMatrix(0, splitIndex - 1, 0, X.columns - 1);
    const Y_train = Y.subMatrix(0, splitIndex - 1, 0, Y.columns - 1);

    const X_test = X.subMatrix(splitIndex, X.rows - 1, 0, X.columns - 1);
    const Y_test = Y.subMatrix(splitIndex, Y.rows - 1, 0, Y.columns - 1);

    console.log('X_train dimensions:', X_train.rows, X_train.columns);
    console.log('Y_train dimensions:', Y_train.rows, Y_train.columns);
    console.log('X_test dimensions:', X_test.rows, X_test.columns);
    console.log('Y_test dimensions:', Y_test.rows, Y_test.columns);

    const regression = new MultivariateLinearRegression(X_train, Y_train);

    const Y_pred = regression.predict(X_test);

    console.log('Predicted values:', Y_pred);

    if (Y_pred && typeof Y_pred[0] === 'number') {
      const Y_pred_matrix = new Matrix([[Y_pred[0]]]);
      const residuals = Y_test.clone().sub(Y_pred_matrix);
      const mse = residuals.pow(2).sum() / residuals.rows;

      const meanY = Y_test.mean('column');
      const totalSumOfSquares = Y_test.clone().sub(meanY).pow(2).sum();
      const rSquared = 1 - (mse * residuals.rows / totalSumOfSquares);

      setMeanSquaredError(mse);
      setRSquared(rSquared);
    } else {
      console.error('Prediction contains non-numeric values:', Y_pred);
    }

    const fetchData = async () => {
      try {
        const responses = await Promise.all([
          fetch(BASE_URL + '/api.php/ventas/total'),
          fetch(BASE_URL + '/api.php/mermas/total'),
          fetch(BASE_URL + '/api.php/oferta/total'),
          fetch(BASE_URL + '/api.php/demanda/total'),
        ]);

        const data = await Promise.all(responses.map((response) => response.json()));

        const latestWastes = parseFloat(data[1][0].total_mermas) || 0;
        const latestOffers = parseFloat(data[2][0].total_oferta_mes) || 0;
        const latestDemands = parseFloat(data[3][0].total_demanda_mes) || 0;

        const normLatestOffers = minMaxScale(latestOffers, Math.min(...offers), Math.max(...offers));
        const normLatestWastes = minMaxScale(latestWastes, Math.min(...wastes), Math.max(...wastes));
        const normLatestDemands = minMaxScale(latestDemands, Math.min(...demands), Math.max(...demands));

        const prediction = regression.predict([normLatestOffers, normLatestWastes, normLatestDemands])[0];

        if (typeof prediction === 'number') {
          const predictionDenorm = denormalize(prediction, Math.min(...sales), Math.max(...sales));
          setPredictionDenormalized(predictionDenorm);
        } else {
          console.error('Prediction contains non-numeric values:', prediction);
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, [data]);

  return (
    <div>
      <h1>Modelo de Regresión Lineal Múltiple</h1>
      <p>{`The predicted amount of sales is: $${predictionDenormalized.toFixed(2)}`}</p>
      <p>{`Mean squared error: ${meanSquaredError.toFixed(4)}`}</p>
      <p>{`R² (Accuracy): ${(rSquared * 100).toFixed(2)}%`}</p>
    </div>
  );
};

export default MultipleLinearRegression;

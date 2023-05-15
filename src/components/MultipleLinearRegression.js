import {Matrix} from 'ml-matrix';
import {useState, useEffect} from 'react';
import MultivariateLinearRegression from 'ml-regression-multivariate-linear';
import ExtractCoefficient from '../utils/ExtractCoefficient';

const MultipleLinearRegression = ({data}) => {
  const [predictionDenormalized, setPredictionDenormalized] = useState(0);
  const [meanSquaredError, setMeanSquaredError] = useState(0);
  const [accuracy, setAccuracy] = useState(0);

  useEffect(() => {
    const {offers, wastes, demands, sales} = data;

    const minOffer = Math.min(...offers);
    const maxOffer = Math.max(...offers);
    const normOffers = offers.map(
      (offer) => (offer - minOffer) / (maxOffer - minOffer)
    );

    const normWastes = wastes.map(
      (w) => (w - wastes[0]) / (wastes[wastes.length - 1] - wastes[0])
    );
    const normDemands = demands.map(
      (d) => (d - demands[0]) / (demands[demands.length - 1] - demands[0])
    );

    const normSales = sales.map(
      (s) => (s - sales[0]) / (sales[sales.length - 1] - sales[0])
    );

    const testData = new Matrix([
      normSales,
      normWastes,
      normDemands,
      normOffers,
    ]).transpose();

    const outputs = new Matrix([normSales]).transpose();

    const regression = new MultivariateLinearRegression(testData, outputs);
    const prediction = regression.predict([1, 1, 1, 1])[0];
    const predictionDenorm =
      prediction * (sales[sales.length - 1] - sales[0]) + sales[0];

    const yHat = regression.predict(testData);
    const residuals = outputs.sub(yHat);
    const residualsArray = residuals.to1DArray();
    const residualsSquaredArray = residualsArray.map((residual) =>
      Math.pow(residual, 2)
    );
    const residualsSquaredSum = residualsSquaredArray.reduce(
      (accumulator, currentValue) => accumulator + currentValue,
      0
    );
    const mse = residualsSquaredSum / residualsArray.length;

    const explainedSumOfSquares =
      outputs.to1DArray().reduce((a, b) => a + b, 0) -
      Math.pow(
        outputs.to1DArray().reduce((a, b) => a + b, 0),
        2
      ) /
        outputs.to1DArray().length;
    const totalSumOfSquares =
      residualsSquaredSum +
      explainedSumOfSquares +
      Math.pow(
        outputs.to1DArray().reduce((a, b) => a + b, 0),
        2
      ) /
        outputs.to1DArray().length;
    const accuracy =
      (1 - residualsSquaredSum / totalSumOfSquares) * 100 -
      ExtractCoefficient(mse);

    setPredictionDenormalized(predictionDenorm);
    setMeanSquaredError(mse);
    setAccuracy(accuracy); // Update the state variable name here
  }, [data]);

  return (
    <div>
      <h1> Modelo de Regresion Lineal Multiple </h1>
      <p>{`The predicted amount of sales is: $${predictionDenormalized}`}</p>
      <p>{`Mean squared error: ${meanSquaredError}`}</p>
      <p>{`Accuracy: ${accuracy} `}</p>
    </div>
  );
};

export default MultipleLinearRegression;

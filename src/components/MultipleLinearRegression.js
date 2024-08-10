import { Matrix } from 'ml-matrix';
import { useState, useEffect } from 'react';
import MultivariateLinearRegression from 'ml-regression-multivariate-linear';
import ExtractCoefficient from '../utils/ExtractCoefficient';
import { BASE_URL } from '../utils/Constants';

const MultipleLinearRegression = ({ data }) => {
  const [predictionDenormalized, setPredictionDenormalized] = useState(0);
  const [meanSquaredError, setMeanSquaredError] = useState(0);
  const [accuracy, setAccuracy] = useState(0);

  useEffect(() => {
    const { offers, wastes, demands, sales } = data;

    const lengthCheck = [offers, wastes, demands, sales].every(
      (arr) => arr.length === offers.length
    );

    if (!lengthCheck) {
      console.error('Input arrays must have the same length');
      return;
    }

    const normalize = (arr) => {
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      if (max === min) {
        return arr.map(() => 0.5);
      }
      return arr.map((value) => (value - min) / (max - min));
    };

    const normOffers = normalize(offers);
    const normWastes = normalize(wastes);
    const normDemands = normalize(demands);
    const normSales = normalize(sales);

    const testData = new Matrix([
      normOffers,
      normWastes,
      normDemands,
      normSales,
    ]).transpose();

    const outputs = new Matrix([normSales]).transpose();

    const regression = new MultivariateLinearRegression(testData, outputs);

    const fetchData = async () => {
      try {
        const responses = await Promise.all([
          fetch(BASE_URL + '/api.php/ventas/total'),
          fetch(BASE_URL + '/api.php/mermas/total'),
          fetch(BASE_URL + '/api.php/oferta/total'),
          fetch(BASE_URL + '/api.php/demanda/total'),
        ]);

        const data = await Promise.all(responses.map((response) => response.json()));

        const latestSales = parseFloat(data[0][0].total_ventas_mes) || 0;
        const latestWastes = parseFloat(data[1][0].total_mermas) || 0;
        const latestOffers = parseFloat(data[2][0].total_oferta_mes) || 0;
        const latestDemands = parseFloat(data[3][0].total_demanda_mes) || 0;

        const normLatestOffers = (latestOffers - Math.min(...offers)) / (Math.max(...offers) - Math.min(...offers));
        const normLatestWastes = (latestWastes - Math.min(...wastes)) / (Math.max(...wastes) - Math.min(...wastes));
        const normLatestDemands = (latestDemands - Math.min(...demands)) / (Math.max(...demands) - Math.min(...demands));
        const normLatestSales = (latestSales - Math.min(...sales)) / (Math.max(...sales) - Math.min(...sales));

        const prediction = regression.predict([normLatestOffers, normLatestWastes, normLatestDemands, normLatestSales])[0];
        const predictionDenorm = prediction * (sales[sales.length - 1] - sales[0]) + sales[0];

        const yHat = regression.predict(testData);
        const residuals = outputs.sub(yHat);
        const residualsArray = residuals.to1DArray();
        const residualsSquaredArray = residualsArray.map((residual) => Math.pow(residual, 2));
        const residualsSquaredSum = residualsSquaredArray.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
        const mse = residualsSquaredSum / residualsArray.length;

        const explainedSumOfSquares = outputs.to1DArray().reduce((a, b) => a + b, 0) - Math.pow(outputs.to1DArray().reduce((a, b) => a + b, 0), 2) / outputs.to1DArray().length;
        const totalSumOfSquares = residualsSquaredSum + explainedSumOfSquares + Math.pow(outputs.to1DArray().reduce((a, b) => a + b, 0), 2) / outputs.to1DArray().length;
        const accuracy = (1 - residualsSquaredSum / totalSumOfSquares) * 100 - ExtractCoefficient(mse);

        setPredictionDenormalized(predictionDenorm);
        setMeanSquaredError(mse);
        setAccuracy(accuracy);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, [data]);

  return (
    <div>
      <h1>Modelo de Regresion Lineal Multiple</h1>
      <p>{`The predicted amount of sales is: $${predictionDenormalized}`}</p>
      <p>{`Mean squared error: ${meanSquaredError}`}</p>
      <p>{`Accuracy: ${accuracy}`}</p>
    </div>
  );
};

export default MultipleLinearRegression;

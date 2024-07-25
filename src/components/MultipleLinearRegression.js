import { Matrix } from 'ml-matrix';
import { useState, useEffect } from 'react';
import MultivariateLinearRegression from 'ml-regression-multivariate-linear';
import ExtractCoefficient from '../utils/ExtractCoefficient';

const MultipleLinearRegression = ({ data }) => {
  const [predictionDenormalized, setPredictionDenormalized] = useState(0);
  const [meanSquaredError, setMeanSquaredError] = useState(0);
  const [accuracy, setAccuracy] = useState(0);

  useEffect(() => {
    const { offers, wastes, demands, sales } = data;

    // Check if all arrays have the same length
    const lengthCheck = [offers, wastes, demands, sales].every(
      (arr) => arr.length === offers.length
    );

    if (!lengthCheck) {
      console.error('Input arrays must have the same length');
      return;
    }

    // Normalization function
    const normalize = (arr) => {
      const min = Math.min(...arr);
      const max = Math.max(...arr);
      return arr.map((value) => (value - min) / (max - min));
    };

    // Normalize all arrays
    const normOffers = normalize(offers);
    const normWastes = normalize(wastes);
    const normDemands = normalize(demands);
    const normSales = normalize(sales);

    // Create the testData and outputs matrices
    const testData = new Matrix([
      normOffers,
      normWastes,
      normDemands,
      normSales,
    ]).transpose();

    const outputs = new Matrix([normSales]).transpose();

    // Perform the regression
    const regression = new MultivariateLinearRegression(testData, outputs);

    // Fetch the latest values from the endpoints
    const fetchData = async () => {
      try {
        const responses = await Promise.all([
          fetch('https://franz.kvconsult.com/fapi-dev/api.php/ventas/total'),
          fetch('https://franz.kvconsult.com/fapi-dev/api.php/mermas/total'),
          fetch('https://franz.kvconsult.com/fapi-dev/api.php/oferta/total'),
          fetch('https://franz.kvconsult.com/fapi-dev/api.php/demanda/total'),
        ]);

        const data = await Promise.all(responses.map((response) => response.json()));

        const latestSales = parseFloat(data[0][0].total_ventas_mes);
        const latestWastes = parseFloat(data[1][0].total_mermas);
        const latestOffers = parseFloat(data[2][0].total_oferta_mes);
        const latestDemands = parseFloat(data[3][0].total_demanda_mes);

        // Normalize the latest values
        const normLatestOffers = (latestOffers - Math.min(...offers)) / (Math.max(...offers) - Math.min(...offers));
        const normLatestWastes = (latestWastes - Math.min(...wastes)) / (Math.max(...wastes) - Math.min(...wastes));
        const normLatestDemands = (latestDemands - Math.min(...demands)) / (Math.max(...demands) - Math.min(...demands));
        const normLatestSales = (latestSales - Math.min(...sales)) / (Math.max(...sales) - Math.min(...sales));

        // Predict using the latest normalized values
        const prediction = regression.predict([normLatestOffers, normLatestWastes, normLatestDemands, normLatestSales])[0];
        const predictionDenorm = prediction * (sales[sales.length - 1] - sales[0]) + sales[0];

        // Calculate residuals and mean squared error
        const yHat = regression.predict(testData);
        const residuals = outputs.sub(yHat);
        const residualsArray = residuals.to1DArray();
        const residualsSquaredArray = residualsArray.map((residual) => Math.pow(residual, 2));
        const residualsSquaredSum = residualsSquaredArray.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
        const mse = residualsSquaredSum / residualsArray.length;

        // Calculate accuracy
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

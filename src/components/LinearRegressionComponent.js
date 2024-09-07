import React, { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

// API endpoint
const API_URL = 'https://franz.kvconsult.com/fapi-dev/data.php/api';

const LinearRegressionComponent = () => {
  const [predictedSales, setPredictedSales] = useState(null);
  const [mape, setMape] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [mae, setMae] = useState(null);
  const [mse, setMse] = useState(null);
  const [rmse, setRmse] = useState(null);
  const [salesData, setSalesData] = useState([]);

  // Fetch data from the API
  const fetchData = async () => {
    try {
      const response = await fetch(API_URL);
      const data = await response.json();

      // Extract 'ventas_x_dia' (sales per day)
      const sales = data.ventas_x_dia.map(d => d.total);
      setSalesData(sales);
    } catch (error) {
      console.error('Error fetching data from API:', error);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    if (!salesData || salesData.length < 2) {
      return;
    }

    const sales = salesData;
    const time = Array.from({ length: sales.length }, (_, i) => i + 1);

    const maxTime = Math.max(...time);
    const maxSales = Math.max(...sales);
    const xs = tf.tensor2d(time.map(t => t / maxTime), [time.length, 1]);
    const ys = tf.tensor2d(sales.map(s => s / maxSales), [sales.length, 1]);

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    model.compile({
      optimizer: tf.train.sgd(0.01),
      loss: 'meanSquaredError',
    });

    async function trainModel() {
      await model.fit(xs, ys, {
        epochs: 200,
        validationSplit: 0.2,
      });

      // Predicted and actual values
      const predictedYs = model.predict(xs).mul(maxSales).dataSync();
      const actualYs = sales;

      // MAPE (Mean Absolute Percentage Error)
      const mape = tf.metrics.meanAbsolutePercentageError(
        tf.tensor2d(actualYs, [actualYs.length, 1]),
        tf.tensor2d(predictedYs, [predictedYs.length, 1])
      ).dataSync()[0];
      setMape(mape);
      setAccuracy(100 - mape);

      // MAE (Mean Absolute Error)
      const mae = tf.metrics.meanAbsoluteError(
        tf.tensor2d(actualYs, [actualYs.length, 1]),
        tf.tensor2d(predictedYs, [predictedYs.length, 1])
      ).dataSync()[0];
      setMae(mae);

      // MSE (Mean Squared Error)
      const mse = tf.metrics.meanSquaredError(
        tf.tensor2d(actualYs, [actualYs.length, 1]),
        tf.tensor2d(predictedYs, [predictedYs.length, 1])
      ).dataSync()[0];
      setMse(mse);

      // RMSE (Root Mean Squared Error)
      const rmse = Math.sqrt(mse);
      setRmse(rmse);

      // Predict next time point
      const nextTimePoint = (sales.length + 1) / maxTime;
      const predictedSalesTensor = model.predict(
        tf.tensor2d([nextTimePoint], [1, 1])
      ).mul(maxSales);
      const predictedSales = predictedSalesTensor.dataSync()[0];
      setPredictedSales(Math.max(0, predictedSales));
    }

    trainModel();
  }, [salesData]);

  if (!salesData || salesData.length < 2) {
    return <div>Insufficient data to perform linear regression.</div>;
  }

  return (
    <div>
      <h1>Modelo de Regresión Lineal Simple</h1>
      {predictedSales !== null && (
        <>
          <p>Predicted sales for the next period: {predictedSales.toFixed(2)}</p>
          <p>Mean Absolute Percentage Error (MAPE): {mape.toFixed(2)}%</p>
          <p>Accuracy: {accuracy.toFixed(2)}%</p>
          <p>Mean Absolute Error (MAE): {mae.toFixed(4)}</p>
          <p>Mean Squared Error (MSE): {mse.toFixed(4)}</p>
          <p>Root Mean Squared Error (RMSE): {rmse.toFixed(4)}</p>
        </>
      )}
    </div>
  );
};

export default LinearRegressionComponent;

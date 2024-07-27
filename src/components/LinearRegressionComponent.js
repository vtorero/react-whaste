import React, { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

const LinearRegressionComponent = ({ data }) => {
  const [predictedSales, setPredictedSales] = useState(null);
  const [mape, setMape] = useState(null);
  const [accuracy, setAccuracy] = useState(null);

  useEffect(() => {
    if (!data.sales || data.sales.length < 2) {
      return;
    }

    const sales = data.sales;
    const time = Array.from({ length: sales.length }, (_, i) => i + 1);

    // Normalize the data
    const maxTime = Math.max(...time);
    const maxSales = Math.max(...sales);
    const xs = tf.tensor2d(time.map(t => t / maxTime), [time.length, 1]);
    const ys = tf.tensor2d(sales.map(s => s / maxSales), [sales.length, 1]);

    // Define the model
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    // Compile the model
    model.compile({
      optimizer: tf.train.sgd(0.01),
      loss: 'meanAbsoluteError',
    });

    // Train the model
    async function trainModel() {
      await model.fit(xs, ys, {
        epochs: 200,
        validationSplit: 0.2,
      });

      // Evaluate the model
      const predictedYs = model.predict(xs).mul(maxSales).dataSync();
      const mape = tf.metrics.meanAbsolutePercentageError(
        tf.tensor2d(sales, [sales.length, 1]),
        tf.tensor2d(predictedYs, [predictedYs.length, 1])
      ).dataSync()[0];
      setMape(mape);
      setAccuracy(100 - mape);

      // Predict the next value
      const nextTimePoint = (sales.length + 1) / maxTime;
      const predictedSalesTensor = model.predict(
        tf.tensor2d([nextTimePoint], [1, 1])
      ).mul(maxSales);
      const predictedSales = predictedSalesTensor.dataSync()[0];
      setPredictedSales(Math.max(0, predictedSales)); // Ensure non-negative prediction
    }

    trainModel();
  }, [data]);

  if (!data.sales || data.sales.length < 2) {
    return <div>Insufficient data to perform linear regression.</div>;
  }

  return (
    <div>
      <h1>Modelo de Regresión Lineal Simple</h1>
      {predictedSales !== null && (
        <>
          <p>Predicted sales for the next month: {predictedSales.toFixed(2)}</p>
          <p>Mean Absolute Percentage Error: {mape.toFixed(2)}%</p>
          <p>Accuracy: {accuracy.toFixed(2)}%</p>
        </>
      )}
    </div>
  );
};

export default LinearRegressionComponent;

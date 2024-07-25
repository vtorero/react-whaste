import * as knnClassifier from '@tensorflow-models/knn-classifier';
import * as tf from '@tensorflow/tfjs';
import { useEffect, useState } from 'react';

const Clasificacion = ({ data }) => {
  const [mse, setMse] = useState(null);
  const [r2, setR2] = useState(null);
  const [predictedSales, setPredictedSales] = useState(null);

  useEffect(() => {
    async function run() {
      const classifier = knnClassifier.create();
      const sales = data.sales;
      const tensorSales = tf.tensor1d(sales);

      const { mean, variance } = tf.moments(tensorSales);
      const std = tf.sqrt(variance);
      const normTensorSales = tensorSales.sub(mean).div(std);

      for (let i = 0; i < sales.length - 1; i++) {
        classifier.addExample(
          normTensorSales.slice([i], [1]),
          normTensorSales.slice([i + 1], [1])
        );
      }

      const newSale = 80;
      const normNewSale = tf.scalar(newSale).sub(mean).div(std);
      const prediction = await classifier.predictClass(normNewSale);
      const denormalizedPrediction = std
        .mul(tf.tensor1d([prediction.confidences[prediction.label]]))
        .add(mean);

      setPredictedSales(denormalizedPrediction.dataSync()[0]);

      const predictions = [];
      for (let i = 0; i < sales.length - 1; i++) {
        const prediction = await classifier.predictClass(
          normTensorSales.slice([i], [1])
        );
        const denormalizedPrediction = std
          .mul(tf.tensor1d([prediction.confidences[prediction.label]]))
          .add(mean);
        predictions.push(denormalizedPrediction.dataSync()[0]);
      }

      const mseTensor = tf.losses.meanSquaredError(sales.slice(1), predictions);
      const mseValue = await mseTensor.data();
      setMse(mseValue);

      const totalSumSquares = tf
        .sum(tf.squaredDifference(tensorSales.mean(), tensorSales))
        .dataSync()[0];
      const residualSumSquares = tf
        .sum(
          tf.squaredDifference(
            tensorSales.slice([1], [sales.length - 1]),
            predictions
          )
        )
        .dataSync()[0];
      const r2Value = 1 - residualSumSquares / totalSumSquares;
      setR2(r2Value);
    }

    run();
  }, [data.sales]);

  return (
    <div>
      <h1>Modelo de Clasificación</h1>
      <p>MSE: {mse}</p>
      <p>R2: {r2}</p>
      <p>
        Predicted sales for {80}: {predictedSales}
      </p>
    </div>
  );
};

export default Clasificacion;

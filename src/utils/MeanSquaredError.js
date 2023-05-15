function MeanSquaredError(actual, predicted) {
  const squaredErrors = actual.map((actualValue, index) => {
    const predictedValue = predicted[index];
    return Math.pow(actualValue - predictedValue, 2);
  });
  const sumSquaredErrors = squaredErrors.reduce(
    (sum, squaredError) => sum + squaredError,
    0
  );
  return sumSquaredErrors / actual.length;
}

export default MeanSquaredError;

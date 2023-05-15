import Mean from './Mean';
function CoefficientOfDeterminant(actual, predicted) {
  const actualMean = Mean(actual);
  const totalSumOfSquares = actual.reduce(
    (sum, actualValue) => sum + Math.pow(actualValue - actualMean, 2),
    0
  );
  const residualSumOfSquares = actual.reduce((sum, actualValue, index) => {
    const predictedValue = predicted[index];
    return sum + Math.pow(actualValue - predictedValue, 2);
  }, 0);
  const explainedSumOfSquares = totalSumOfSquares - residualSumOfSquares;
  return explainedSumOfSquares / totalSumOfSquares;
}
export default CoefficientOfDeterminant;

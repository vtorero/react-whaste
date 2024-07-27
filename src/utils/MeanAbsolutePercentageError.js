const MeanAbsolutePercentageError = (actual, predicted) => {
  let total = 0;
  for (let i = 0; i < actual.length; i++) {
    total += Math.abs((actual[i] - predicted[i]) / actual[i]);
  }
  return (total / actual.length) * 100;
};

export default MeanAbsolutePercentageError;

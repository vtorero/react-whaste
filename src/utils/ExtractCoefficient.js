function ExtractCoefficient(number) {
  const str = number.toString();
  const match = str.match(/([-0-9.]+)e/);

  if (!match) {
    return str;
  }

  const coefficient = parseFloat(match[1]);
  return coefficient;
}

export default ExtractCoefficient;

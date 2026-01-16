# weather-robustness-benchmark
Measure how a standard CNN’s performance + confidence changes as weather-like corruptions get worse.

# core deliverables
1. degradation curves (accuracy vs severity)
2. confidence/entropy plots
3. failure examples grid
4. short "takeaways" section

### Blur
| Accuracy                                   | Confidence                                        | Entropy                                        |
| ------------------------------------------ | ------------------------------------------------- | ---------------------------------------------- |
| ![](figures/accuracy_vs_severity_blur.png) | ![](figures/mean_confidence_vs_severity_blur.png) | ![](figures/mean_entropy_vs_severity_blur.png) |

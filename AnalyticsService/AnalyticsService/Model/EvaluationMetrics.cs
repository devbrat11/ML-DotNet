using Microsoft.ML.Data;

namespace AnalyticsService.Model
{
    public class EvaluationMetrics
    {
        
    }

    public class RegressionMetrics : EvaluationMetrics
    {
        public RegressionMetrics(Microsoft.ML.Data.RegressionMetrics metrics)
        {
            RootMeanSquaredError = metrics.RootMeanSquaredError;
        }

        public double RootMeanSquaredError { get; private set; }
    }
}
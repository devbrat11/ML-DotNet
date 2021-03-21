using Microsoft.ML.Data;

namespace MovieRatingPredictor.Model
{
    /// <summary>
    /// Model class for training data.
    /// </summary>
    public class MovieRating
    {
        [LoadColumn(0)]
        public float userId;
        [LoadColumn(1)]
        public float movieId;
        [LoadColumn(2)]
        public float Label;
    }

    /// <summary>
    /// Model class for predicted Rating.
    /// </summary>
    public class MovieRatingPrediction
    {
        public float Label;
        public float Score;
    }
}

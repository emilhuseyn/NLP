using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;

namespace SentimentAnalysis
{
    class Program
    {
        // Defining the SentimentData model
        public class SentimentData
        {
            public string Text { get; set; }
            public bool Label { get; set; }
        }

        // Class for showing prediction results
        public class SentimentPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool Prediction { get; set; }
            public float Probability { get; set; }
            public float Score { get; set; }
        }

        static void Main(string[] args)
        {
            // Create the MLContext
            var mlContext = new MLContext();

            // Training data list
            var trainingData = new List<SentimentData>();

            Console.WriteLine("Enter training sentences and their sentiment (true for positive, false for negative). Type 'done' to finish entering training data.");

            // Collect training data from the user
            while (true)
            {
                // Ask for the sentence
                Console.Write("Enter a sentence (or type 'done' to finish): ");
                string text = Console.ReadLine();

                if (text.ToLower() == "done")
                    break;

                // Ask for the sentiment (true/false)
                Console.Write("Is this sentence positive (true/false): ");
                bool label = bool.Parse(Console.ReadLine());

                // Add the input to training data
                trainingData.Add(new SentimentData { Text = text, Label = label });
            }

            // Load the training data into a DataView
            var trainDataView = mlContext.Data.LoadFromEnumerable(trainingData);

            // Create the pipeline and train the model
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(SentimentData.Label), featureColumnName: "Features"));

            var model = pipeline.Fit(trainDataView);

            Console.WriteLine("Enter a sentence to predict its sentiment (type 'exit' to quit):");

            while (true)
            {
                // Get user input for prediction
                string inputText = Console.ReadLine();

                if (inputText.ToLower() == "exit")
                    break;

                var inputData = new SentimentData { Text = inputText };

                // Create a PredictionEngine
                var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

                // Make the prediction
                var result = predictionEngine.Predict(inputData);

                // Display the result
                Console.WriteLine($"Text: {inputText} | Prediction: {(result.Prediction ? "Positive" : "Negative")} | Probability: {result.Probability}");
                Console.WriteLine();
            }
        }
    }
}

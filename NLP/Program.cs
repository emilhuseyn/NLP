using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace SentimentAnalysis
{
    class Program
    {
        // SentimentData modelimizi yaradırıq
        public class SentimentData
        {
            public string Text { get; set; }

            [LoadColumn(1)]
            public bool Label { get; set; }
        }

        // Nəticəni göstərmək üçün klass
        public class SentimentPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool Prediction { get; set; }
            public float Probability { get; set; }
            public float Score { get; set; }
        }

        static void Main(string[] args)
        {
            // ML konteyneri yaradırıq
            var mlContext = new MLContext();

            // Nümunə məlumatlar (Öyrədilmə üçün)
            var trainingData = new[]
            {
                new SentimentData { Text = "Bu məhsulu çox bəyəndim!", Label = true },
                new SentimentData { Text = "Heç xoşuma gəlmədi.", Label = false },
                new SentimentData { Text = "Bu çox maraqlı idi.", Label = true },
                new SentimentData { Text = "Bu xidmət dəhşətli idi.", Label = false }
            };

             var trainDataView = mlContext.Data.LoadFromEnumerable(trainingData);

             var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(SentimentData.Label), featureColumnName: "Features"));

             var model = pipeline.Fit(trainDataView);

            
            Console.WriteLine("Mətn daxil edin (çıkmaq üçün 'exit' yazın):");

            while (true)
            {
                string inputText = Console.ReadLine();

                if (inputText.ToLower() == "exit")
                    break;

                var inputData = new SentimentData { Text = inputText };

                // Test məlumatını yükləyirik
                var inputDataView = mlContext.Data.LoadFromEnumerable(new[] { inputData });

                // Proqnozlaşdırma aparırıq
                var prediction = model.Transform(inputDataView);

                // Nəticəni götürmək üçün Prediction Engine yaradırıq
                var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

                var result = predictionEngine.Predict(inputData);

                // Nəticənin göstərilməsi
                Console.WriteLine($"Mətn: {inputText} | Nəticə: {(result.Prediction ? "Müsbət" : "Mənfi")} | Ehtimal: {result.Probability}");
                Console.WriteLine();
            }
        }
    }
}

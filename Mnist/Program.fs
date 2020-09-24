open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Transforms


/// The Digit class represents one MNIST digit
[<CLIMutable>]
type Digit = {
    [<LoadColumn(0)>] Number: float32
    [<LoadColumn(1, 784)>][<VectorType(784)>] PixelValues: float32 array
}

/// The DigitPrediction class represents one MNIST digit prediction
[<CLIMutable>]
type DigitPrediction = {
    Score: float32 array
}

let toPath: string seq -> string = Seq.reduce (fun a b -> Path.Join(a, b))

let trainDataPath =
    [Environment.CurrentDirectory; "data"; "mnist_train.csv"] |> toPath
let testDataPath =
    [Environment.CurrentDirectory; "data"; "mnist_test.csv"] |> toPath

[<EntryPoint>]
let main argv =
    let context = MLContext()

    let trainData = context.Data.LoadFromTextFile<Digit>(trainDataPath, hasHeader=true, separatorChar=',')
    let testData = context.Data.LoadFromTextFile<Digit>(testDataPath, hasHeader=true, separatorChar=',')
    
    let pipeline =
        EstimatorChain()
            // map the number column to a key value and store in a label column
            .Append(
                context.Transforms.Conversion.MapValueToKey(
                    "Label",
                    "Number",
                    keyOrdinality=ValueToKeyMappingEstimator.KeyOrdinality.ByValue
                )
            )
            // concatenate all feature columns into a single vector
            .Append(context.Transforms.Concatenate("Features", "PixelValues"))
            // cache data to speed up training
            .AppendCacheCheckpoint(context)
            // train model with SDCA (stochastic dual coordinate ascent)
            .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy())
            // map the label key back to a number
            .Append(context.Transforms.Conversion.MapKeyToValue("Number", "Label"))

    let model = trainData |> pipeline.Fit

    let metrics =
        testData
        |> model.Transform
        |> context.MulticlassClassification.Evaluate

    printfn "Evaluation metrics"
    printfn "  MicroAccuracy:    %f" metrics.MicroAccuracy
    printfn "  MacroAccuracy:    %f" metrics.MacroAccuracy
    printfn "  LogLoss:          %f" metrics.LogLoss
    printfn "  LogLossReduction: %f" metrics.LogLossReduction

    // test a handful of individual digits
    let digits = context.Data.CreateEnumerable(testData, reuseRowObject=false) |> Array.ofSeq
    let testDigits = [digits.[5]; digits.[16]; digits.[28]; digits.[63]; digits.[129]]

    let engine = context.Model.CreatePredictionEngine model

    printfn "Model predictions:"
    printf "  #\t\t"; [0..9] |> Seq.iter (printf "%i\t\t"); printfn ""
    testDigits |> Seq.iter (
        fun digit -> 
            printf "  %i\t" (int digit.Number)
            let prediction = engine.Predict digit
            prediction.Score |> Seq.iter (printf "%f\t")
            printfn ""
    )

    0

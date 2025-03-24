const express = require("express")
const multer = require("multer")
const tf = require("@tensorflow/tfjs-node")
const path = require("path")
const sharp = require("sharp")

const app = express()
const PORT = 9000

const storage = multer.memoryStorage()
const upload = multer({storage})

let model
async function LoadModel(){
    try {
        console.log("Loading model...")
        model = await tf.loadGraphModel('file://' + path.join(__dirname, 'tfjs_model/model.json'))
        console.log("Model loaded")
    } catch (error) {
        console.log("Error loading model:", error)
    }
}

LoadModel()

async function PreprocessImage(imageBuffer){
    try {
        const resizedBuffer = await sharp(imageBuffer).resize(128, 128).toBuffer()
        const tensor = tf.node.decodeImage(resizedBuffer, 3)
        const resizedImage = tf.image.resizeBilinear(tensor, [128, 128])
        const mean = resizedImage.mean()
        const variance = resizedImage.squaredDifference(mean).mean()
        const std = variance.sqrt()
        const adjustedStd = std.maximum(tf.scalar(1.0/Math.sqrt(resizedImage.size)))
        const normalizedImage = resizedImage.sub(mean).div(adjustedStd)
        const batchedImage = normalizedImage.expandDims(0)
        return batchedImage
    } catch (error) {
        console.log("Error preprocessing image:", error)
        throw error
    }
}

app.post('/predict', upload.single('image'), async (req, res) => {
    if (!model) {
        return res.status(503).json({error: 'Model not loaded yet'})
    }
    if (!req.file) {
        return res.status(400).json({error: 'Image file not provided'})
    }
    try {
        const imageBuffer = req.file.buffer
        const preprocessedImage = await PreprocessImage(imageBuffer)
        const prediction = await model.predict(preprocessedImage)
        const predictedClass = prediction.argMax(1).dataSync()[0]
        const dict = {0: 'Cloud', 1: 'Other', 2: 'Smoke'}
        const predictedLabel = dict[predictedClass]
        res.json({predictedClass, predictedLabel})
    } catch (error) {
        console.log("Error predicting image:", error)
        res.status(500).json({error: 'An error occurred during prediction'})  
    }
})

app.use(express.static(__dirname))

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`)
})

    // app.use('/tfjs', express.static(path.join(__dirname, 'node_modules/@tensorflow/tfjs/dist')));
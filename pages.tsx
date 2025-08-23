"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Upload, FileText, Brain, CheckCircle, AlertTriangle, Loader2 } from "lucide-react"

export default function ManoSwastikClassifier() {
  const [file, setFile] = useState<File | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<{
    classification: number
    confidence?: number
    timestamp?: string
  } | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0]
    if (selectedFile) {
      if (selectedFile.name.toLowerCase().endsWith(".edf")) {
        setFile(selectedFile)
        setError(null)
        setResult(null)
      } else {
        setError("Please select a valid .edf file")
        setFile(null)
      }
    }
  }, [])

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
  }, [])

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    const droppedFile = event.dataTransfer.files[0]
    if (droppedFile && droppedFile.name.toLowerCase().endsWith(".edf")) {
      setFile(droppedFile)
      setError(null)
      setResult(null)
    } else {
      setError("Please drop a valid .edf file")
    }
  }, [])

  const handleClassify = async () => {
    if (!file) return

    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append("file", file)

      // Replace with your Flask API endpoint
      const response = await fetch("/api/classify", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Classification failed")
      }

      const data = await response.json()
      setResult({
        classification: data.classification,
        confidence: data.confidence,
        timestamp: new Date().toLocaleString(),
      })
    } catch (err) {
      setError("Failed to classify the EDF file. Please try again.")
      console.error("Classification error:", err)
    } finally {
      setIsLoading(false)
    }
  }

  const getResultDisplay = () => {
    if (!result) return null

    const isHealthy = result.classification === 1
    return {
      label: isHealthy ? "Healthy" : "Schizophrenia",
      color: isHealthy ? "bg-emerald-500" : "bg-red-500",
      icon: isHealthy ? CheckCircle : AlertTriangle,
      description: isHealthy
        ? "The EDF analysis indicates normal brain activity patterns."
        : "The EDF analysis indicates patterns consistent with schizophrenia.",
    }
  }

  const resultDisplay = getResultDisplay()

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center space-x-3">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-3 rounded-xl">
              <Brain className="h-8 w-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                ManoSwastik
              </h1>
              <p className="text-gray-600 font-medium">EDF Neural Classification System</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">Advanced EDF Signal Analysis</h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Upload your EDF (European Data Format) files for AI-powered neural pattern classification and schizophrenia
            detection using cutting-edge machine learning algorithms.
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-2">
          {/* Upload Section */}
          <Card className="border-2 border-dashed border-gray-200 hover:border-blue-300 transition-colors">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Upload className="h-5 w-5 text-blue-600" />
                <span>Upload EDF File</span>
              </CardTitle>
              <CardDescription>Select or drag and drop your .edf file for neural signal analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div
                className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors cursor-pointer"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={() => document.getElementById("file-input")?.click()}
              >
                <input id="file-input" type="file" accept=".edf" onChange={handleFileUpload} className="hidden" />

                {file ? (
                  <div className="space-y-3">
                    <FileText className="h-12 w-12 text-green-600 mx-auto" />
                    <div>
                      <p className="font-semibold text-gray-900">{file.name}</p>
                      <p className="text-sm text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                    <Badge variant="secondary" className="bg-green-100 text-green-800">
                      Ready for Analysis
                    </Badge>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <Upload className="h-12 w-12 text-gray-400 mx-auto" />
                    <div>
                      <p className="text-lg font-semibold text-gray-700">Drop your EDF file here</p>
                      <p className="text-sm text-gray-500">or click to browse files</p>
                    </div>
                  </div>
                )}
              </div>

              {error && (
                <Alert className="mt-4 border-red-200 bg-red-50">
                  <AlertTriangle className="h-4 w-4 text-red-600" />
                  <AlertDescription className="text-red-800">{error}</AlertDescription>
                </Alert>
              )}

              <Button
                onClick={handleClassify}
                disabled={!file || isLoading}
                className="w-full mt-6 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold py-3"
                size="lg"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                    Analyzing Neural Patterns...
                  </>
                ) : (
                  <>
                    <Brain className="h-5 w-5 mr-2" />
                    Classify EDF Signal
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Results Section */}
          <Card className="border-2">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Brain className="h-5 w-5 text-purple-600" />
                <span>Classification Results</span>
              </CardTitle>
              <CardDescription>AI-powered analysis results and confidence metrics</CardDescription>
            </CardHeader>
            <CardContent>
              {result && resultDisplay ? (
                <div className="space-y-6">
                  <div className="text-center">
                    <div
                      className={inline-flex items-center space-x-3 px-6 py-4 rounded-xl ${resultDisplay.color} text-white}
                    >
                      <resultDisplay.icon className="h-8 w-8" />
                      <span className="text-2xl font-bold">{resultDisplay.label}</span>
                    </div>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                    <p className="text-gray-700 text-center">{resultDisplay.description}</p>

                    {result.confidence && (
                      <div className="text-center">
                        <p className="text-sm text-gray-600">Confidence Score</p>
                        <p className="text-2xl font-bold text-gray-900">{(result.confidence * 100).toFixed(1)}%</p>
                      </div>
                    )}

                    {result.timestamp && (
                      <p className="text-xs text-gray-500 text-center">Analyzed on {result.timestamp}</p>
                    )}
                  </div>

                  <Alert className="border-blue-200 bg-blue-50">
                    <AlertTriangle className="h-4 w-4 text-blue-600" />
                    <AlertDescription className="text-blue-800">
                      <strong>Medical Disclaimer:</strong> This tool is for research purposes only. Always consult with
                      qualified healthcare professionals for medical diagnosis.
                    </AlertDescription>
                  </Alert>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <Brain className="h-16 w-16 mx-auto mb-4 text-gray-300" />
                  <p className="text-lg font-medium">No Analysis Yet</p>
                  <p className="text-sm">Upload an EDF file to see classification results</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Info Section */}
        <Card className="mt-12 bg-gradient-to-r from-blue-50 to-purple-50 border-blue-200">
          <CardContent className="pt-6">
            <div className="grid md:grid-cols-3 gap-6 text-center">
              <div>
                <div className="bg-blue-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                  <FileText className="h-6 w-6 text-blue-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">EDF Format Support</h3>
                <p className="text-sm text-gray-600">Supports European Data Format files for neural signal analysis</p>
              </div>
              <div>
                <div className="bg-purple-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                  <Brain className="h-6 w-6 text-purple-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">AI Classification</h3>
                <p className="text-sm text-gray-600">
                  Advanced machine learning algorithms for accurate pattern recognition
                </p>
              </div>
              <div>
                <div className="bg-green-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                  <CheckCircle className="h-6 w-6 text-green-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">Instant Results</h3>
                <p className="text-sm text-gray-600">Get classification results with confidence scores in seconds</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
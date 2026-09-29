import { useEffect, useState } from 'react'

function Recommendations() {
  const [applications, setApplications] = useState([])
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    async function loadRecommendations() {
      try {
        // Get applications from backend
        const applicationsResponse = await fetch(
          'http://127.0.0.1:8000/applications'
        )

        if (!applicationsResponse.ok) {
          throw new Error('Failed to fetch applications')
        }

        const apps = await applicationsResponse.json()
        setApplications(apps)

        // Get recommendation for each application
        const recommendationResults = await Promise.all(
          apps.map(async (app) => {
            const response = await fetch(
              'http://127.0.0.1:8000/recommendation',
              {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  application_id: app.id,
                }),
              }
            )

            if (!response.ok) {
              throw new Error(`Failed for ${app.id}`)
            }

            return response.json()
          })
        )

        setRecommendations(recommendationResults)
        setLoading(false)
      } catch (err) {
        console.error(err)
        setError('Unable to load recommendations from backend')
        setLoading(false)
      }
    }

    loadRecommendations()
  }, [])

  if (loading) {
    return (
      <div className="page">
        <h1>6R Recommendations</h1>
        <p>Loading recommendations...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="page">
        <h1>6R Recommendations</h1>
        <p>{error}</p>
      </div>
    )
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>6R Recommendations</h1>
          <p>
            AI-powered migration strategy recommendations for applications.
          </p>
        </div>

        <div className="application-count">
          {recommendations.length} Analyzed
        </div>
      </div>

      <div className="recommendation-grid">
        {recommendations.map((recommendation) => {
          const application = applications.find(
            (app) => app.id === recommendation.application_id
          )

          return (
            <div
              className="card recommendation-card"
              key={recommendation.application_id}
            >
              <div className="recommendation-top">
                <div>
                  <h2>{application?.name}</h2>
                  <span>{recommendation.application_id}</span>
                </div>

                <span className="strategy-badge">
                  {recommendation.recommendation}
                </span>
              </div>

              <div className="confidence">
                <div className="confidence-header">
                  <span>AI Confidence</span>

                  <strong>
                    {Math.round(recommendation.confidence * 100)}%
                  </strong>
                </div>

                <div className="confidence-bar">
                  <div
                    style={{
                      width: `${recommendation.confidence * 100}%`,
                    }}
                  ></div>
                </div>
              </div>

              <div className="explanation">
                <strong>Why this recommendation?</strong>
                <p>{recommendation.explanation}</p>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default Recommendations
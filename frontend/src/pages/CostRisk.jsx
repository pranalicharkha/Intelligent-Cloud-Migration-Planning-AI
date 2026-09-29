import { useEffect, useState } from 'react'

function CostRisk() {
  const [applications, setApplications] = useState([])
  const [costRisks, setCostRisks] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    async function loadCostRisk() {
      try {
        // Get applications
        const applicationsResponse = await fetch(
          'http://127.0.0.1:8000/applications'
        )

        if (!applicationsResponse.ok) {
          throw new Error('Failed to fetch applications')
        }

        const apps = await applicationsResponse.json()
        setApplications(apps)

        // Get cost and risk for each application
        const results = await Promise.all(
          apps.map(async (app) => {
            const response = await fetch(
              'http://127.0.0.1:8000/cost-risk',
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

        setCostRisks(results)
        setLoading(false)
      } catch (err) {
        console.error(err)
        setError('Unable to load cost and risk data from backend')
        setLoading(false)
      }
    }

    loadCostRisk()
  }, [])

  const getApplicationName = (id) => {
    const application = applications.find(
      (app) => app.id === id
    )

    return application ? application.name : id
  }

  const getRiskClass = (score) => {
    if (score >= 70) return 'high'
    if (score >= 40) return 'medium'
    return 'low'
  }

  if (loading) {
    return (
      <div className="page">
        <h1>Cost & Risk</h1>
        <p>Loading cost and risk data...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="page">
        <h1>Cost & Risk</h1>
        <p>{error}</p>
      </div>
    )
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Cost & Risk</h1>
          <p>
            Estimated cloud migration cost and application risk analysis.
          </p>
        </div>

        <div className="application-count">
          {costRisks.length} Applications
        </div>
      </div>

      <div className="cost-risk-grid">
        {costRisks.map((item) => (
          <div
            className="card cost-risk-card"
            key={item.application_id}
          >
            <div className="cost-risk-header">
              <div>
                <h2>{getApplicationName(item.application_id)}</h2>
                <span>{item.application_id}</span>
              </div>

              <span
                className={`risk-score ${getRiskClass(
                  item.risk_score
                )}`}
              >
                Risk {item.risk_score}
              </span>
            </div>

            <div className="cost-section">
              <p>Monthly AWS Cost</p>

              <h3>
                ${item.monthly_aws_cost.toLocaleString()}
              </h3>
            </div>

            <div className="cost-range">
              <div>
                <span>Lower Estimate</span>
                <strong>
                  ${item.cost_range.lower.toLocaleString()}
                </strong>
              </div>

              <div>
                <span>Upper Estimate</span>
                <strong>
                  ${item.cost_range.upper.toLocaleString()}
                </strong>
              </div>
            </div>

            <div className="risk-bar-section">
              <div className="risk-bar-header">
                <span>Risk Score</span>
                <strong>{item.risk_score}/100</strong>
              </div>

              <div className="risk-bar">
                <div
                  className={getRiskClass(item.risk_score)}
                  style={{
                    width: `${item.risk_score}%`,
                  }}
                ></div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default CostRisk
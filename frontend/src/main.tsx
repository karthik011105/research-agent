import { StrictMode, Suspense, lazy } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import './index.css'
import { Layout } from './components/Layout'
import { ResearchPage } from './pages/ResearchPage'
import { HistoryDetailPage, HistoryPage } from './pages/HistoryPage'
import { WatchPage } from './pages/WatchPage'
// the results page bundles the charts and result data: loaded only when opened
const DashboardPage = lazy(() => import('./pages/DashboardPage').then(m => ({ default: m.DashboardPage })))

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<ResearchPage />} />
          <Route path="history" element={<HistoryPage />} />
          <Route path="history/:jobId" element={<HistoryDetailPage />} />
          <Route path="watch" element={<WatchPage />} />
          <Route path="dashboard" element={<Suspense fallback={null}><DashboardPage /></Suspense>} />
          <Route path="*" element={<ResearchPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)

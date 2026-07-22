/**
 * The instrument chassis.
 *
 * A fixed frame rather than a scrolling marketing page: a persistent status
 * rail on the left carrying live device and model state, and a working bench on
 * the right where modules dock. Routes cross-fade rather than snapping, and the
 * active-nav marker slides between items so navigation reads as movement within
 * one instrument rather than a series of page loads.
 */

import { Suspense, lazy, useCallback, useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { NavLink, Navigate, Outlet, Route, Routes, useLocation } from 'react-router-dom';

import { StatusRail } from './components/StatusRail';
import { ThemeToggle, type Theme } from './components/ThemeToggle';
import { pageVariants } from './motion/transitions';
import { api, type Health } from './lib/api';

const Bench = lazy(() => import('./views/Bench').then((m) => ({ default: m.Bench })));
const Live = lazy(() => import('./views/Live').then((m) => ({ default: m.Live })));
const Enroll = lazy(() => import('./views/Enroll').then((m) => ({ default: m.Enroll })));
const SpeakerMapView = lazy(() => import('./views/SpeakerMapView').then((m) => ({ default: m.SpeakerMapView })));
const Arena = lazy(() => import('./views/Arena').then((m) => ({ default: m.Arena })));
const Lab = lazy(() => import('./views/Lab').then((m) => ({ default: m.Lab })));
const HowItWorks = lazy(() => import('./views/HowItWorks').then((m) => ({ default: m.HowItWorks })));
const Landing = lazy(() => import('./views/Landing').then((m) => ({ default: m.Landing })));

const ROUTES = [
  { to: '/analyse', label: 'Analyse', hint: 'Record or upload a voice' },
  { to: '/live', label: 'Live', hint: 'Who is speaking now' },
  { to: '/enrol', label: 'Enrol', hint: 'Teach it a new speaker' },
  { to: '/map', label: 'Map', hint: 'The embedding space' },
  { to: '/lab', label: 'Lab', hint: 'Break it on purpose' },
  { to: '/arena', label: 'Arena', hint: 'Model comparison' },
  { to: '/how-it-works', label: 'How', hint: 'From sound to identity' },
];

function readTheme(): Theme {
  const attr = document.documentElement.getAttribute('data-theme');
  return attr === 'print' ? 'print' : 'scope';
}

export function App() {
  const [theme, setTheme] = useState<Theme>(readTheme);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const location = useLocation();

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('linguistix-theme', theme);
    } catch {
      // Private browsing blocks storage; the theme still applies for this session.
    }
  }, [theme]);

  const refreshHealth = useCallback(async () => {
    try {
      setHealth(await api.health());
      setHealthError(null);
    } catch (error) {
      setHealthError(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  return (
    <>
      {/* Grain and vignette over the body gradient. Static, composited once. */}
      <div className="backdrop" aria-hidden="true">
        <div className="backdrop__grain" />
        <div className="backdrop__vignette" />
      </div>

      <Routes location={location}>
        {/* Full bleed: no rail, no status column. */}
        <Route
          path="/"
          element={
            <Suspense fallback={<div className="loading eyebrow">Loading…</div>}>
              <Landing theme={theme} />
            </Suspense>
          }
        />

        {/* Everything else renders inside the instrument chassis. */}
        <Route
          element={
            <Chassis
              theme={theme}
              onTheme={setTheme}
              health={health}
              healthError={healthError}
              onRetry={refreshHealth}
            />
          }
        >
          <Route path="/analyse" element={<Bench theme={theme} health={health} />} />
          <Route path="/live" element={<Live theme={theme} />} />
          <Route path="/enrol" element={<Enroll theme={theme} onChanged={refreshHealth} />} />
          <Route path="/map" element={<SpeakerMapView theme={theme} />} />
          <Route path="/lab" element={<Lab theme={theme} />} />
          <Route path="/arena" element={<Arena theme={theme} />} />
          <Route path="/how-it-works" element={<HowItWorks />} />
          <Route path="*" element={<Navigate to="/analyse" replace />} />
        </Route>
      </Routes>
    </>
  );
}

interface ChassisProps {
  theme: Theme;
  onTheme: (theme: Theme) => void;
  health: Health | null;
  healthError: string | null;
  onRetry: () => void;
}

/** The instrument frame: rail on the left, routed bench on the right. */
function Chassis({ theme, onTheme, health, healthError, onRetry }: ChassisProps) {
  const location = useLocation();

  return (
    <div className="chassis">
      <a className="skip-link" href="#bench">
        Skip to the instrument
      </a>

      <header className="chassis__rail" role="banner">
        <div className="brand">
          {/* The mark is a link home now that there is somewhere to go. */}
          <NavLink to="/" className="brand__home" aria-label="Linguistix home">
            <span className="brand__mark" aria-hidden="true" />
          </NavLink>
          <span className="brand__name">Linguistix</span>
          <span className="brand__sub eyebrow">Sonagraph</span>
        </div>

        <nav aria-label="Sections">
          <ul className="nav">
            {ROUTES.map((route) => (
              <li key={route.to}>
                <NavLink to={route.to} className={({ isActive }) => `nav__link${isActive ? ' is-active' : ''}`}>
                  {({ isActive }) => (
                    <>
                      {isActive ? (
                        <motion.span
                          className="nav__marker"
                          layoutId="nav-marker"
                          transition={{ type: 'spring', stiffness: 400, damping: 34 }}
                        />
                      ) : null}
                      <span className="nav__label">{route.label}</span>
                      <span className="nav__hint">{route.hint}</span>
                    </>
                  )}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        <StatusRail health={health} error={healthError} onRetry={onRetry} />
        <ThemeToggle theme={theme} onChange={onTheme} />
      </header>

      <main className="chassis__bench" id="bench">
        <Suspense fallback={<div className="loading eyebrow">Loading module…</div>}>
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              variants={pageVariants}
              initial="hidden"
              animate="visible"
              exit="exit"
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </Suspense>
      </main>
    </div>
  );
}

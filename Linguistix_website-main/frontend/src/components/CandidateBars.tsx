/**
 * Ranked candidates with the decision threshold drawn in.
 *
 * The threshold line is the point: a top score of 0.44 next to a threshold of
 * 0.39 is a very different result from the same score against a threshold of
 * 0.60, and a bar chart without it hides that.
 */

import type { Candidate } from '../lib/api';
import { confidenceColor } from '../lib/colormap';

interface Props {
  candidates: Candidate[];
  threshold: number;
  theme: 'scope' | 'print';
}

/** Cosine runs -1..1; map to 0..1 so bars share one scale with the threshold. */
const toFraction = (score: number) => (score + 1) / 2;

export function CandidateBars({ candidates, threshold, theme }: Props) {
  if (!candidates.length) return null;
  const thresholdPct = toFraction(threshold) * 100;

  return (
    <ol className="bars">
      {candidates.map((candidate) => {
        const fraction = toFraction(candidate.score);
        const accepted = candidate.score >= threshold;
        return (
          <li className="bar" key={`${candidate.source}:${candidate.speaker}`}>
            <span className="bar__name" title={candidate.speaker}>
              {candidate.speaker}
              {candidate.source === 'enrolled' ? <span className="bar__tag"> enrolled</span> : null}
            </span>
            <span className="bar__track">
              <span
                className="bar__fill"
                style={{
                  width: `${Math.max(0, fraction * 100)}%`,
                  background: confidenceColor(fraction, theme),
                  opacity: accepted ? 1 : 0.45,
                }}
              />
              <span className="bar__threshold" style={{ left: `${thresholdPct}%` }} aria-hidden="true" />
            </span>
            <span className="bar__value">{candidate.similarity_pct.toFixed(1)}%</span>
          </li>
        );
      })}
    </ol>
  );
}

// "One GPU, a whole endoscopy department": the real DeepStream 9.1 tiled output of 12 colonoscopy rooms (YOLO26 on
// TensorRT, NvDCF), with per-room alarm glows from the logged detections, plus room 13 = the robot endoscope rendered by
// NVIDIA Isaac Sim at the same time on the same RTX 5060 Ti. 4:5, 30 fps.
import React from 'react';
import {AbsoluteFill, Img, OffthreadVideo, interpolate, staticFile, useCurrentFrame, Easing} from 'remotion';
import {loadFont as loadOnest} from '@remotion/google-fonts/Onest';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';
import data from '../public/center_data.json';
import robot from '../public/robot.json';

const {fontFamily: SANS} = loadOnest('normal', {weights: ['400', '600', '800'], subsets: ['latin']});
const {fontFamily: MONO} = loadMono('normal', {weights: ['400', '500'], subsets: ['latin']});

type Alarm = {t0: number; t1: number; true: boolean};
type Room = {room: number; video: string; alarms: Alarm[]; caught: string[]; polyps: string[]};
const D = data as unknown as {rooms: Room[]; totals: {polyps: number; caught: number; alarms: number; false_alarms: number};
  stats: {fps_per_room: number; total_fps: number}; bench: {fps_per_room: number}};
const R = robot as unknown as {path: {t: number; s: 'N' | 'A' | 'I'; c: number; b: number[] | null}[]; finds: {t: number; class: string; size_mm: number}[]};

export const W = 1080, H = 1350, FPS = 30;
const INTRO = 3.5, MAIN = 24, OUTRO = 6;
export const TOTAL = Math.round((INTRO + MAIN + OUTRO) * FPS);
const LIVE = INTRO + MAIN;                                       // wall plays 1x for this long
const ROBOT_SPEED = (R.path.length / FPS) / LIVE;                // fit the whole robot run into the live part
const VRAM_GB = 4.7, GPU_UTIL = 94;                               // measured: nvidia-smi while all 13 rooms ran
const BG = '#05070C', TEXT = '#F4F1EA', DIM = '#8A929C', NV = '#76B900', AMBER = '#FFB020', RED = '#FF4D4D';
const CLAMP = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
const OUT = Easing.bezier(0.16, 1, 0.3, 1);
const ramp = (f: number, a: number, b: number) => interpolate(f, [a * FPS, b * FPS], [0, 1], {...CLAMP, easing: OUT});
const NAME: Record<string, string> = {AD: 'adenoma', HP: 'hyperplastic', SSL: 'serrated lesion'};

const WALL_Y = 172, WALL_H = 648, TILE_W = W / 4, TILE_H = WALL_H / 3;

const Wall: React.FC = () => {
  const f = useCurrentFrame();
  const t = f / FPS;
  const tilt = (1 - ramp(f, 0.3, 2.6)) * 22;
  const scale = interpolate(ramp(f, 0.3, 2.6), [0, 1], [0.86, 1]);
  return (
    <div style={{position: 'absolute', left: 0, top: WALL_Y, width: W, height: WALL_H, perspective: 1400}}>
      <div style={{position: 'absolute', inset: 0, transform: `rotateX(${tilt}deg) scale(${scale})`, transformOrigin: '50% 100%'}}>
        <OffthreadVideo src={staticFile('wall.mp4')} muted style={{width: W, height: WALL_H}} />
        {D.rooms.map((r, i) => {
          const a = r.alarms.find((x) => t >= x.t0 && t <= x.t1 + 0.4);
          const col = a ? (a.true ? NV : RED) : 'rgba(255,255,255,0.10)';
          return (
            <div key={r.room} style={{position: 'absolute', left: (i % 4) * TILE_W, top: Math.floor(i / 4) * TILE_H, width: TILE_W, height: TILE_H,
              boxSizing: 'border-box', border: `${a ? 4 : 1}px solid ${col}`, boxShadow: a ? `inset 0 0 26px ${col}` : 'none'}}>
              <div style={{position: 'absolute', left: 8, top: 7, fontFamily: MONO, fontSize: 15, color: TEXT, padding: '2px 8px', borderRadius: 8,
                background: 'rgba(5,7,12,0.7)'}}>ROOM {String(r.room).padStart(2, '0')}</div>
              {a && <div style={{position: 'absolute', right: 8, top: 7, fontFamily: MONO, fontSize: 15, color: '#071000', padding: '2px 8px',
                borderRadius: 8, background: col}}>{a.true ? 'POLYP' : 'CHECK'}</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
};

const RobotRoom: React.FC = () => {
  const f = useCurrentFrame();
  const i = Math.min(R.path.length - 1, Math.floor(f * ROBOT_SPEED));
  const p = R.path[i];
  const w = 520, h = 416, k = w / 640;
  const col = p.s === 'I' ? NV : p.s === 'A' ? AMBER : '#7DB7FF';
  const showBox = p.b && (p.c >= 0.45 || p.s !== 'N');
  const found = R.finds.filter((x) => x.t <= p.t).length;
  return (
    <div style={{position: 'absolute', left: 24, top: WALL_Y + WALL_H + 34, width: w, height: h, borderRadius: 18, overflow: 'hidden',
      border: `3px solid ${col}`, boxShadow: `0 0 30px ${col}55`}}>
      <Img src={staticFile(`robot/${String(i).padStart(5, '0')}.jpg`)} style={{width: w, height: h}} />
      {showBox && p.b && <div style={{position: 'absolute', left: p.b[0] * k, top: p.b[1] * k, width: (p.b[2] - p.b[0]) * k, height: (p.b[3] - p.b[1]) * k,
        border: `4px solid ${col}`, borderRadius: 10, boxShadow: `0 0 18px ${col}`}} />}
      <div style={{position: 'absolute', left: 10, top: 9, fontFamily: MONO, fontSize: 18, color: TEXT, padding: '4px 10px', borderRadius: 9,
        background: 'rgba(5,7,12,0.75)'}}>ROOM 13 · ROBOT · ISAAC SIM</div>
      <div style={{position: 'absolute', left: 10, bottom: 9, fontFamily: MONO, fontSize: 18, color: col, padding: '4px 10px', borderRadius: 9,
        background: 'rgba(5,7,12,0.75)'}}>{p.s === 'I' ? 'INSPECT' : p.s === 'A' ? 'APPROACH' : 'NAVIGATE'} · found {found}/3</div>
    </div>
  );
};

const Kpi: React.FC<{label: string; value: string; color?: string; y: number}> = ({label, value, color = TEXT, y}) => (
  <div style={{position: 'absolute', left: 572, top: y, width: 480}}>
    <div style={{fontFamily: MONO, fontSize: 20, color: DIM, letterSpacing: 1}}>{label}</div>
    <div style={{fontFamily: SANS, fontWeight: 800, fontSize: 58, letterSpacing: -2, color, lineHeight: 1.05}}>{value}</div>
  </div>
);

const Panel: React.FC = () => {
  const f = useCurrentFrame();
  const t = f / FPS;
  const o = ramp(f, INTRO - 0.6, INTRO);
  const alarms = D.rooms.reduce((n, r) => n + r.alarms.filter((a) => a.t0 <= t).length, 0);
  const y0 = WALL_Y + WALL_H + 30;
  return (
    <div style={{opacity: o}}>
      <Kpi label="ROOMS LIVE" value="12 + 1 robot" y={y0} />
      <Kpi label="EACH ROOM" value={`${D.stats.fps_per_room.toFixed(0)} fps · real time`} color={NV} y={y0 + 108} />
      <Kpi label="WHOLE GPU" value={`${VRAM_GB} GB · ${GPU_UTIL}%`} y={y0 + 216} />
      <Kpi label="AI ALARMS SO FAR" value={String(alarms)} color={AMBER} y={y0 + 324} />
    </div>
  );
};

const Header: React.FC = () => {
  const f = useCurrentFrame();
  const intro = 1 - ramp(f, INTRO - 0.5, INTRO);
  return (
    <>
      {intro > 0.01 && (
        <div style={{position: 'absolute', left: 48, right: 48, top: 26, opacity: intro, fontFamily: SANS, fontWeight: 800, fontSize: 54, lineHeight: 1.04,
          letterSpacing: -2.5, color: TEXT, zIndex: 5}}>
          One GPU.<span style={{color: NV}}> A whole endoscopy department.</span>
        </div>
      )}
      <div style={{position: 'absolute', left: 48, top: 110, opacity: 1 - intro, fontFamily: MONO, fontSize: 24, color: TEXT, display: 'flex', gap: 14}}>
        <span style={{color: NV}}>● LIVE</span> DeepStream 9.1 · TensorRT · YOLO26 · Isaac Sim · one RTX 5060 Ti
      </div>
    </>
  );
};

const Outro: React.FC = () => {
  const f = useCurrentFrame();
  const o = ramp(f, LIVE, LIVE + 0.8);
  if (o < 0.01) return null;
  const T = D.totals;
  return (
    <AbsoluteFill style={{opacity: o, background: `rgba(5,7,12,${0.95 * o})`}}>
      <div style={{position: 'absolute', left: 64, right: 64, top: 220, fontFamily: SANS, fontWeight: 800, fontSize: 96, letterSpacing: -3, color: TEXT, lineHeight: 1}}>
        13 rooms.<div style={{color: NV}}>One RTX 5060 Ti.</div>
      </div>
      <div style={{position: 'absolute', left: 64, right: 64, top: 500, fontFamily: MONO, fontSize: 28, lineHeight: 1.75, color: TEXT}}>
        12 real colonoscopies · {D.stats.fps_per_room.toFixed(1)} fps each, together with<br />a robot endoscope in Isaac Sim · {VRAM_GB} GB VRAM in total<br />
        {T.caught} of {T.polyps} polyps flagged · {T.false_alarms} false alarms in 12 min
      </div>
      <div style={{position: 'absolute', left: 64, right: 64, top: 790, fontFamily: MONO, fontSize: 23, lineHeight: 1.7, color: DIM}}>
        DeepStream 9.1 (GStreamer) · nvstreammux batch 12 · TensorRT FP16<br />Ultralytics YOLO26 · NvDCF tracker · NVIDIA Isaac Sim 6.2<br />
        7 of the 12 videos were in the detector's training set: this is about throughput
      </div>
      <div style={{position: 'absolute', left: 64, right: 64, bottom: 80, fontFamily: MONO, fontSize: 22, lineHeight: 1.6, color: DIM}}>
        Research demo · not a medical device · REAL-Colon (CC BY 4.0)<br />Zalkarbek Arapbaev · physician & CV engineer
      </div>
    </AbsoluteFill>
  );
};

export const Center: React.FC = () => (
  <AbsoluteFill style={{background: BG}}>
    <Wall />
    <RobotRoom />
    <Panel />
    <Header />
    <Outro />
  </AbsoluteFill>
);

import React from 'react';
import {Composition} from 'remotion';
import {Center, TOTAL, W, H, FPS} from './Center';

export const Root: React.FC = () => (
  <Composition id="Center" component={Center} durationInFrames={TOTAL} fps={FPS} width={W} height={H} />
);

import { Easing } from 'react-native';
import tokens from '../tokens/brand-tokens.json';

type Bezier = [number, number, number, number];

const std: Bezier = (tokens.motion?.curve?.standard as Bezier) ?? [0.2, 0, 0, 1];
const emp: Bezier = (tokens.motion?.curve?.emphasized as Bezier) ?? [0.2, 0, 0, 1];

export const motion = {
  dur: tokens.motion?.dur ?? { xs: 120, sm: 200, md: 320, lg: 420 },
  curve: {
    standard: Easing.bezier(...std),
    emphasized: Easing.bezier(...emp),
  },
};

use anyhow::Result;

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmFormat {
    Raw,
    Alaw,
    Ulaw,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Pcm { sample_rate: Option<usize>, format: PcmFormat },
    Wav,
    OggOpus,
}

impl std::str::FromStr for Format {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let format = match s.to_lowercase().as_str() {
            "" | "pcm" => Default::default(),
            "pcm_8000" => Self::pcm(8000),
            "pcm_16000" => Self::pcm(16000),
            "pcm_22050" => Self::pcm(22050),
            "pcm_24000" => Self::pcm(24000),
            "pcm_44100" => Self::pcm(44100),
            "pcm_48000" => Self::pcm(48000),
            "ulaw_8000" => Self::ulaw(8000),
            "mulaw_8000" => Self::ulaw(8000),
            "alaw_8000" => Self::alaw(8000),
            "wav" => Self::Wav,
            "opus" => Self::OggOpus,
            s => anyhow::bail!(
                "unsupported output format '{s}', supported formats: 'pcm', 'pcm_24000', 'ulaw_8000', 'alaw_8000', 'wav', 'opus'"
            ),
        };
        Ok(format)
    }
}

impl Format {
    pub fn pcm(sample_rate: usize) -> Self {
        Self::Pcm { sample_rate: Some(sample_rate), format: PcmFormat::Raw }
    }

    pub fn ulaw(sample_rate: usize) -> Self {
        Self::Pcm { sample_rate: Some(sample_rate), format: PcmFormat::Ulaw }
    }

    pub fn alaw(sample_rate: usize) -> Self {
        Self::Pcm { sample_rate: Some(sample_rate), format: PcmFormat::Alaw }
    }
}

impl Default for Format {
    fn default() -> Self {
        Self::Pcm { sample_rate: None, format: PcmFormat::Raw }
    }
}

pub enum Decoder {
    OggOpus(kaudio::ogg_opus::Decoder),
    Pcm { fft: Option<(Vec<f32>, Box<rubato::FftFixedOut<f32>>)>, format: PcmFormat },
    Wav(crate::wav::Decoder),
}

impl Decoder {
    pub fn new(format: Format, out_sample_rate: usize, frame_size: usize) -> Result<Self> {
        match format {
            Format::OggOpus => Self::ogg_opus(out_sample_rate, frame_size),
            Format::Pcm { sample_rate, format } => {
                let sample_rate = sample_rate.unwrap_or(out_sample_rate);
                let fft = if sample_rate == out_sample_rate {
                    None
                } else {
                    use rubato::Resampler;
                    let fft = rubato::FftFixedOut::<f32>::new(
                        sample_rate,
                        out_sample_rate,
                        frame_size,
                        1,
                        1,
                    )?;
                    let buf: Vec<f32> = Vec::with_capacity(fft.input_frames_next());
                    Some((buf, Box::new(fft)))
                };
                Ok(Self::Pcm { fft, format })
            }
            Format::Wav => Ok(Self::wav(out_sample_rate, frame_size)?),
        }
    }

    fn ogg_opus(sample_rate: usize, frame_size: usize) -> Result<Self> {
        Ok(Self::OggOpus(kaudio::ogg_opus::Decoder::new(sample_rate, frame_size)?))
    }

    fn wav(sample_rate: usize, frame_size: usize) -> Result<Self> {
        let decoder = crate::wav::Decoder::new(sample_rate, frame_size)?;
        Ok(Self::Wav(decoder))
    }

    pub fn decode(&mut self, data: &[u8]) -> Result<Vec<f32>> {
        let pcm = match self {
            Self::OggOpus(oo) => match oo.decode(data)? {
                None => vec![],
                Some(pcm) => pcm.to_vec(),
            },
            Self::Wav(decoder) => decoder.decode(data)?,
            Self::Pcm { fft, format } => {
                // TODO(laurent): this is inefficient as it's doing intermediate copies.
                use byteorder::ByteOrder;
                if !data.len().is_multiple_of(2) {
                    anyhow::bail!("pcm data length is not a multiple of 2 {}", data.len());
                }
                let pcm: Vec<f32> = match format {
                    PcmFormat::Raw => data
                        .chunks_exact(2)
                        .map(|b| {
                            let v = byteorder::LittleEndian::read_i16(b);
                            v as f32 / i16::MAX as f32
                        })
                        .collect(),
                    PcmFormat::Alaw => data
                        .iter()
                        .map(|&s| law_decoder::alaw_decode_sample(s) as f32 / i16::MAX as f32)
                        .collect(),
                    PcmFormat::Ulaw => data
                        .iter()
                        .map(|&s| law_decoder::ulaw_decode_sample(s) as f32 / i16::MAX as f32)
                        .collect(),
                };
                match fft {
                    Some((buf, fft)) => {
                        use rubato::Resampler;
                        let mut pcm_out = vec![];
                        buf.extend_from_slice(&pcm);
                        while buf.len() >= fft.input_frames_next() {
                            let input: Vec<f32> = buf.drain(..fft.input_frames_next()).collect();
                            let pcm_resampled = fft.process(&[&input], None)?;
                            match pcm_resampled.into_iter().next() {
                                None => anyhow::bail!("resampling produced no output"),
                                Some(pcm_resampled) => pcm_out.extend_from_slice(&pcm_resampled),
                            }
                        }
                        pcm_out
                    }
                    None => pcm,
                }
            }
        };
        Ok(pcm)
    }
}

mod law_decoder {
    // https://github.com/foss-for-synopsys-dwc-arc-processors/G722/blob/293bd03a21f6ce0adeddf1ef541e0ddc18fea5fc/g711/g711.c#L164
    pub fn alaw_decode_sample(a_val: u8) -> i16 {
        let a_val = a_val ^ 0x55;
        let t = a_val as i16 & 0x0F;
        let seg = (a_val & 0x70) >> 4;
        let t = if seg != 0 { (t + t + 1 + 32) << (seg + 2) } else { (t + t + 1) << 3 };
        if a_val & 0x80 != 0 { t } else { -t }
    }

    // https://github.com/foss-for-synopsys-dwc-arc-processors/G722/blob/293bd03a21f6ce0adeddf1ef541e0ddc18fea5fc/g711/g711.c#L298
    pub fn ulaw_decode_sample(input: u8) -> i16 {
        let u_val = !input;
        let t = ((u_val as i16 & 0x0f) << 3) + 0x84;
        let t = t << ((u_val as i16 & 0x70) >> 4);
        if u_val & 0x80 != 0 { 0x84 - t } else { t - 0x84 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resampling_pcm_decoder() -> Result<()> {
        let format = Format::Pcm { sample_rate: Some(8000), format: PcmFormat::Raw };
        let mut decoder = Decoder::new(format, 24000, 1920)?;
        let pcm_8k: Vec<f32> =
            (0..8000).map(|i| ((i as f32 / 8000.0) * 2.0 * std::f32::consts::PI).sin()).collect();
        use byteorder::ByteOrder;
        let mut buf = vec![0u8; pcm_8k.len() * 2];
        let pcm_i16: Vec<i16> = pcm_8k.iter().map(|&v| (v * i16::MAX as f32) as i16).collect();
        byteorder::LittleEndian::write_i16_into(&pcm_i16, &mut buf);
        let pcm_decoded = decoder.decode(&buf)?;
        assert_eq!(pcm_decoded.len(), 23040);
        // 40ms of audio at 8kHz = 320 samples so 640 bytes
        let pcm_decoded = decoder.decode(&[0u8; 640])?;
        assert_eq!(pcm_decoded.len(), 1920);
        for _ in 0..10 {
            let pcm_decoded = decoder.decode(&[0u8; 640])?;
            assert_eq!(pcm_decoded.len(), 0);
            let pcm_decoded = decoder.decode(&[0u8; 640])?;
            assert_eq!(pcm_decoded.len(), 1920);
        }
        Ok(())
    }

    #[test]
    fn test_alaw_decode_sample() {
        let samples = (0u8..=255).map(law_decoder::alaw_decode_sample).collect::<Vec<i16>>();
        assert_eq!(
            samples,
            [
                -5504, -5248, -6016, -5760, -4480, -4224, -4992, -4736, -7552, -7296, -8064, -7808,
                -6528, -6272, -7040, -6784, -2752, -2624, -3008, -2880, -2240, -2112, -2496, -2368,
                -3776, -3648, -4032, -3904, -3264, -3136, -3520, -3392, -22016, -20992, -24064,
                -23040, -17920, -16896, -19968, -18944, -30208, -29184, -32256, -31232, -26112,
                -25088, -28160, -27136, -11008, -10496, -12032, -11520, -8960, -8448, -9984, -9472,
                -15104, -14592, -16128, -15616, -13056, -12544, -14080, -13568, -344, -328, -376,
                -360, -280, -264, -312, -296, -472, -456, -504, -488, -408, -392, -440, -424, -88,
                -72, -120, -104, -24, -8, -56, -40, -216, -200, -248, -232, -152, -136, -184, -168,
                -1376, -1312, -1504, -1440, -1120, -1056, -1248, -1184, -1888, -1824, -2016, -1952,
                -1632, -1568, -1760, -1696, -688, -656, -752, -720, -560, -528, -624, -592, -944,
                -912, -1008, -976, -816, -784, -880, -848, 5504, 5248, 6016, 5760, 4480, 4224,
                4992, 4736, 7552, 7296, 8064, 7808, 6528, 6272, 7040, 6784, 2752, 2624, 3008, 2880,
                2240, 2112, 2496, 2368, 3776, 3648, 4032, 3904, 3264, 3136, 3520, 3392, 22016,
                20992, 24064, 23040, 17920, 16896, 19968, 18944, 30208, 29184, 32256, 31232, 26112,
                25088, 28160, 27136, 11008, 10496, 12032, 11520, 8960, 8448, 9984, 9472, 15104,
                14592, 16128, 15616, 13056, 12544, 14080, 13568, 344, 328, 376, 360, 280, 264, 312,
                296, 472, 456, 504, 488, 408, 392, 440, 424, 88, 72, 120, 104, 24, 8, 56, 40, 216,
                200, 248, 232, 152, 136, 184, 168, 1376, 1312, 1504, 1440, 1120, 1056, 1248, 1184,
                1888, 1824, 2016, 1952, 1632, 1568, 1760, 1696, 688, 656, 752, 720, 560, 528, 624,
                592, 944, 912, 1008, 976, 816, 784, 880, 848
            ]
        );
    }

    #[test]
    fn test_ulaw_decode_sample() {
        let samples = (0u8..=255).map(law_decoder::ulaw_decode_sample).collect::<Vec<i16>>();
        assert_eq!(
            samples,
            [
                -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956, -23932, -22908,
                -21884, -20860, -19836, -18812, -17788, -16764, -15996, -15484, -14972, -14460,
                -13948, -13436, -12924, -12412, -11900, -11388, -10876, -10364, -9852, -9340,
                -8828, -8316, -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140, -5884, -5628,
                -5372, -5116, -4860, -4604, -4348, -4092, -3900, -3772, -3644, -3516, -3388, -3260,
                -3132, -3004, -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980, -1884, -1820,
                -1756, -1692, -1628, -1564, -1500, -1436, -1372, -1308, -1244, -1180, -1116, -1052,
                -988, -924, -876, -844, -812, -780, -748, -716, -684, -652, -620, -588, -556, -524,
                -492, -460, -428, -396, -372, -356, -340, -324, -308, -292, -276, -260, -244, -228,
                -212, -196, -180, -164, -148, -132, -120, -112, -104, -96, -88, -80, -72, -64, -56,
                -48, -40, -32, -24, -16, -8, 0, 32124, 31100, 30076, 29052, 28028, 27004, 25980,
                24956, 23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764, 15996, 15484, 14972,
                14460, 13948, 13436, 12924, 12412, 11900, 11388, 10876, 10364, 9852, 9340, 8828,
                8316, 7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140, 5884, 5628, 5372, 5116, 4860,
                4604, 4348, 4092, 3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004, 2876, 2748, 2620,
                2492, 2364, 2236, 2108, 1980, 1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436, 1372,
                1308, 1244, 1180, 1116, 1052, 988, 924, 876, 844, 812, 780, 748, 716, 684, 652,
                620, 588, 556, 524, 492, 460, 428, 396, 372, 356, 340, 324, 308, 292, 276, 260,
                244, 228, 212, 196, 180, 164, 148, 132, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48,
                40, 32, 24, 16, 8, 0
            ]
        );
    }
}

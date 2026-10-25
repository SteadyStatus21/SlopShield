// Demo feed contents. File name prefix is the ground truth used by scripts/eval.py:
// ai_*.mp4 = AI-generated, real_*.mp4 = real footage.
// Clips come from the held-out benchmark (bench/vidtest, never trained on); the files themselves are
// gitignored, so copy them in with the mapping below. Captions are written for the demo.
// Mix of cases: AI with a platform label, AI with only hashtags, AI with no text cues (vision has to
// catch it), and real clips, including a cartoon and one with a caption that mentions AI.
//
//   ai_01 = ai_kling_video_000101   ai_02 = ai_veo31_veo31_16_0   ai_03 = ai_veo31_veo31_75_0
//   ai_04 = ai_veo31_veo31_55_0     ai_05 = ai_sora_017_20250114_sora
//   real_01 = real_youtube_-qGd1JwYLbQ   real_02 = real_youtube_-urYwYLrcMA   real_03 = real_youtube_-Ar7D-GARhI
//   real_04 = real_youtube_-SD9DkKyOrY   real_05 = real_youtube_-oYYvzarYDs   real_06 = real_youtube_-ECK_BisOLM
window.CLIPS = [
  { file: 'real_01.mp4', author: 'Linda Park', caption: 'Biscuit is worn out after our morning walk 🐶' },
  { file: 'ai_01.mp4', author: 'Faith & Blessings Daily', caption: 'Good morning ☕ God bless your day! Type AMEN and share with 10 friends 🙏', platformLabel: 'AI info' },
  { file: 'real_02.mp4', author: 'Tom Okafor', caption: 'Someone thinks it is dinner time 😹' },
  { file: 'ai_02.mp4', author: 'Nature Wonders', caption: 'Rare footage of a snow leopard in the Himalayas 😮 #sora #aivideo' },
  { file: 'real_03.mp4', author: 'Westside Farmers Market', caption: 'Meet Dolly, the newest member of our farm family 🐑 Come say hi Saturday!' },
  { file: 'ai_03.mp4', author: 'Yellowstone Wildlife Fans', caption: 'A lone wolf spotted at dawn this morning. Nature is incredible 🐺' },
  { file: 'real_04.mp4', author: 'Maria Gonzalez', caption: "My brother's band played their first gig tonight 🥁 so proud!" },
  { file: 'ai_04.mp4', author: 'Crypto Wealth Tips', caption: 'Our new trading app turned $250 into $9,800 in one week 💰 Only 20 spots left, link in comments 👇' },
  { file: 'real_05.mp4', author: 'Ridge Car Club', caption: 'Track day! Mike finally got his BMW out on the Nürburgring 🏁 (no, this is not AI lol)' },
  { file: 'ai_05.mp4', author: 'Kai Mahoe', caption: 'Dawn patrol at Pipeline 🌊🏄 what a morning' },
  { file: 'real_06.mp4', author: 'Little Sprouts Preschool', caption: "The kids voted: Mr. Broccoli is the star of this week's veggie song 🥦" },
];

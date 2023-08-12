# Face Stabilizer

Very WIP

## Create a video from the generated frames

`exa --no-icons` works instead of `ls -v`

```console
$ ls -v | cat -n | while read n f; do mv -n "$f" "$n.jpg"; done
$ ffmpeg -framerate 30 -pattern_type sequence -start_number 1 -r 3 -i %d.jpg -s 1080x1920 out.mp4
```

# TeenLyrics: Multi-Label Explicit Content Detection with Age-Based Rating
This repository contains a complete pipeline for multi-label explicit content detection in music lyrics, culminating in an automatic MCR rating with content descriptors.

## Overview
We developed a deep learning model that detects multiple explicit content categories in lyrics simultaneously. The system assigns probability scores for each category and maps them to a final MCR rating.

## Features

## MCR Rating Levels
| Rating | Meaning |
|--------|---------|
| **M-E** (Everyone) | Suitable for all ages; no explicit sexual themes, violence, substance use, or strong language. |
| **M-P** (Parental Guidance Suggested) | Some material may not be suitable for children; may contain mild language, minimal suggestive themes, or very brief non-graphic violence. |
| **M-T** (Teen) | Suitable for ages 13 and up; may contain violence, suggestive themes, drug references, or infrequent strong language. |
| **M-R** (Restricted) | Under 17 requires adult guardian; may contain intense violence, strong sexual content, frequent strong language, or drug abuse. |
| **M-AO** (Adults Only) | Suitable only for adults 18+; may contain graphic sexual content, extreme violence, or glorified drug use. |

## Usage
